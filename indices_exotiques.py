"""Une fonction pour évaluer les indicateurs simples de mutualisation 
   sur un portefeuille assurantiel
"""

# Quelques chargements de packages
import numpy as np
import pandas as pd
import math

### Machine Learning Models
import lightgbm as lgbm


### Kolmogorov complexity modules
from pybdm import BDM, ctmdata, PartitionIgnore, PartitionRecursive, PartitionCorrelated
import pybdm

class LgbmBasedSplitterTree:
    """ a usefull class to encode actuarial portofolio, base on LightGBM
    """
    # change num_leaves to have a less depth splitter 
    def __init__(self, num_leaves=10000):
        self.params = {
            'objective': 'regression',
            'bagging_freq': 0,
            'max_depth':-1,
            'num_boost_round':1,
            'num_leaves': num_leaves,
            "random_state":0,
        }

    # train the splitter    
    def train(self, lgbm_train):
        self.gbm = lgbm.train( params=self.params, train_set=lgbm_train)
    
    # use the splitter to predict portofolio
    def predict(self, X):
        return self.gbm.predict(X)
    
    # use to transform splitter as dataframe 
    def get_tree_as_df(self):
        self.tree_df = self.gbm.trees_to_dataframe()
        self.tree_df = self.tree_df.set_index("node_index")
        return self.tree_df
    
    # get sub partitions under the given partition node
    def get_child_nodes(self, node):
        if self.tree_df.loc[node, 'left_child']==None:
            return (None, None)
        else:
            return (self.tree_df.loc[node, 'left_child'], self.tree_df.loc[node, 'right_child'])

    # get all terminal leaves under the given partition node
    def get_terminal_leaf(self, node):
        if self.tree_df.loc[node, 'left_child']==None:
            return [node]
        else:
            return self.get_terminal_leaf(self.tree_df.loc[node, 'left_child']) + self.get_terminal_leaf(self.tree_df.loc[node, 'right_child'])

    # recursively compute informations for all nodes in the splitter
    def compute_nodes_infos(self):
        tree_df = self.tree_df.copy()
        split_char = '0-S'
        leaf_char = '0-L'

        # Extracting split sequence base on node index
        split_sequence = list(tree_df[tree_df.index.str.contains(split_char)].index)
        split_sequence = sorted(split_sequence, key=lambda x: int(x[x.find(split_char)+len(split_char):]))
        self.split_sequence = split_sequence
        self.nodes_infos = {}
        self.current_measure_teriminal_nodes = {}
        self.current_measure_teriminal_nodes[0] = ['0-S0']
        market_nb_assures = self.tree_df.loc['0-S0', 'count']

        for d, node in enumerate(split_sequence):
            self.nodes_infos[node] = {}
            child_nodes = self.get_child_nodes(node)
            self.nodes_infos[node]['left_child'] = child_nodes[0]
            self.nodes_infos[node]['rigth_child'] = child_nodes[1]

            terimnal_leaves = self.get_terminal_leaf(node)
            self.nodes_infos[node]['terminal_leaves'] = terimnal_leaves
            self.nodes_infos[node]['terminal_leaves_nb'] = len(terimnal_leaves)
            self.nodes_infos[node]['terminal_leaf_probs'] = [self.tree_df.loc[ter_node, 'count'] / market_nb_assures for ter_node in terimnal_leaves]

            # Residual heterogeneity in the leaves under node
            props_to_ones = np.array(self.nodes_infos[node]['terminal_leaf_probs']) / sum(self.nodes_infos[node]['terminal_leaf_probs'])
            self.nodes_infos[node]['unit_entropy'] = - props_to_ones @ np.log2(props_to_ones)
            self.nodes_infos[node]['unit_entropy_normalized'] = self.nodes_infos[node]['unit_entropy'] / np.log2(len(terimnal_leaves))

            # Identifie current terminal leaves
            temp = self.current_measure_teriminal_nodes[d].copy()
            temp.remove(node)

            self.current_measure_teriminal_nodes[d + 1] = temp + list(child_nodes)
        
        # Add information about terminal leaves...
        terimal_leaves = list(tree_df[tree_df.index.str.contains(leaf_char)].index)
        self.terimnal_leaves = terimal_leaves
        for d, leaf in enumerate(terimal_leaves):
            self.nodes_infos[leaf] = {}
            child_nodes = self.get_child_nodes(leaf)
            self.nodes_infos[leaf]['left_child'] = None
            self.nodes_infos[leaf]['rigth_child'] = None

            self.nodes_infos[leaf]['terminal_leaves'] = [leaf]
            self.nodes_infos[leaf]['terminal_leaves_nb'] = 1
            self.nodes_infos[leaf]['terminal_leaf_probs'] = [1.0]
            
            # Residual heterogeneity in the leaves under current node
            props_to_ones = [1.0]
            self.nodes_infos[leaf]['unit_entropy'] = 0.0
            self.nodes_infos[leaf]['unit_entropy_normalized'] = 0.0

# Function to encode portfolio

def get_decision_rule(node: str, tree_df: pd.DataFrame, tree_dict: dict, data: pd.DataFrame):
    """Returns for a given node, a boolean vector whose length is data.shape[0]
       this vector indicates whether or not a policyholder passes through the considered node

    Args:
        node (str): node index given as string. ex: '0-S0'
        tree_df (pd.DataFrame): LgbmBasedSplitterTree.tree_df after calling the method get_tree_as_df on LgbmBasedSplitterTree objet
        tree_dict (dict): nodes informtions spliter 
        data (pd.DataFrame): insured protofolio

    Returns:
        np.array: array of bool, indicates for each insured if he satified the node condition or not 
    """
    parent_node = tree_df.loc[node, 'parent_index']

    is_left_child = tree_df.loc[parent_node, 'left_child'] == node
    tresh = tree_df.loc[parent_node, 'threshold']
    if tree_df.loc[parent_node, 'decision_type']=="==":    
        list_of_mod = [int(item) for item in tresh.split('||')]
        last_rule = data[tree_df.loc[parent_node, 'split_feature']].isin(list_of_mod)
    else:
        last_rule = data[tree_df.loc[parent_node, 'split_feature']] <= np.float(tresh)
            
    previous_rules = tree_dict[parent_node]['decision_rule']
    dr = np.logical_and(previous_rules, last_rule) if is_left_child else np.logical_and(previous_rules, np.logical_not(last_rule))
    return dr.values


def encode_portofolio_as_dict(portofolio:pd.DataFrame, splitter:LgbmBasedSplitterTree):
    """_summary_

    Args:
        portofolio (pd.DataFrame): insured protofolio
        splitter (LgbmBasedSplitterTree): _description_

    Returns:
        dict: _description_
    """
    ## Recodification des colonnes catégorielles  ... 
    cat_cols = list(portofolio.select_dtypes(include=['category']).columns)
    portofolio_copy = portofolio.copy()
    for col in cat_cols:
        portofolio_copy[col] = portofolio_copy[col].cat.codes  
        
    ## Macro information sur le marché 
    splitter_df = splitter.tree_df.copy()
    #market_nb_assures = splitter_df.loc['0-S0', 'count']
    
    pf_dict_encoded = {}
    
    #Initialisation for fist split : node 0-S0
    first_node = '0-S0'
    pf_dict_encoded[first_node] = {}
    pf_dict_encoded[first_node]['decision_rule'] = np.array([True]*portofolio.shape[0])
    
    #Loop for others smplit nodes ... 
    for node in splitter.split_sequence[1:]:       
        dr = get_decision_rule(node, splitter_df, pf_dict_encoded, portofolio_copy)
        pf_dict_encoded[node] = {}
        pf_dict_encoded[node]['decision_rule'] = dr

    # Loop over terminal leaves ... 
    for leaf in splitter.terimnal_leaves :
        dr = get_decision_rule(leaf, splitter_df, pf_dict_encoded, portofolio_copy)
        pf_dict_encoded[leaf] = {}
        pf_dict_encoded[leaf]['decision_rule'] = dr
        
    return pf_dict_encoded


def build_pf_description_freq_dec(portofolio, pf_dict_encoded, splitter, precision=4, nb_desc=-1, previous_descp=None, prop_ralative=False):
    """ Builds an encoding of the portfolio distribution such as: 0.13|0.586|0.113|0.255|0.191|0.11|0.062 ... 
        proportions calculated can be absolute (relative to the portfolio) or relative (relative to the parent node)...

    Args:
        portofolio (pd.DataFrame): insured protofolio
        pf_dict_encoded (dict): output of the function encode_portofolio_as_dict
        splitter (LgbmBasedSplitterTree): a trained splitter 
        precision (int, optional): precision order for description. Defaults to 4.
        nb_desc (int, optional): number of measures to be performed on the portfolio. 
                                 Defaults to -1 meaning untill the end of the splitter.
        previous_descp (dict, optional): previous description sequence, to avoid recomputing it over and over. Defaults to None.
        prop_ralative (bool, optional): how measurement should be return, relative or absolute. Defaults to False.

    Returns:
        str: portofolio frequence description
    """
    # get split sequence from the splitter
    split_sequence = splitter.split_sequence
    
    # check if previous description is provided
    if previous_descp!= None:
        # perform a single measure and bind it to previous description
        descp = previous_descp
        split_node = split_sequence[len(previous_descp.split('|'))]
        child_node = splitter.nodes_infos[split_node]['left_child']
        if prop_ralative:
            current_desc = pf_dict_encoded[child_node]['decision_rule'].sum() / pf_dict_encoded[split_node]['decision_rule'].sum()
        else:
            current_desc = pf_dict_encoded[child_node]['decision_rule'].sum() / portofolio.shape[0]
        descp = descp + '|' + str(current_desc.round(precision))
        return descp

    # set nb_split correctly      
    if nb_desc==-1:
        nb_desc = len(split_sequence)
    else:
        nb_desc = min(nb_desc, len(split_sequence))       
    descp = ''

    # perform all measures base on pf_dict_encoded
    for split_node in split_sequence[:nb_desc]:
        child_node = splitter.nodes_infos[split_node]['left_child']
        if prop_ralative: 
            current_desc = pf_dict_encoded[child_node]['decision_rule'].sum() / pf_dict_encoded[split_node]['decision_rule'].sum()
        else:
            current_desc = pf_dict_encoded[child_node]['decision_rule'].sum() / portofolio.shape[0]
        descp = descp + str(current_desc.round(precision)) + '|'
    return descp[:-1]

def build_pf_description_OL_dec(portofolio, pf_dict_encoded, splitter, precision=4, nb_desc=-1, previous_descp=None, gain_relatif=False):
    """ Builds an encoding of the optimisation layer volatility applied to portfolio as: 0.3359|0.1436|0.0884|0.1026|0.1165|0.2252|0.1109 
        proportions calculated can be absolute (relative to the portfolio) or relative (relative to the parent node)...

    Args:
        portofolio (pd.DataFrame): insured protofolio
        pf_dict_encoded (dict): output of the function encode_portofolio_as_dict
        splitter (LgbmBasedSplitterTree): a trained splitter 
        precision (int, optional): precision order for description. Defaults to 4.
        nb_desc (int, optional): number of measures to be performed on the portfolio. 
                                 Defaults to -1 meaning untill the end of the splitter.
        previous_descp (dict, optional): previous description sequence, to avoid recomputing it over and over. Defaults to None.
        gain_relatif (bool, optional): how measurement should be return, relative or absolute. Defaults to False.
                                        Each measurement of  gain of homogeny that can be in vision :
                                        > Absolute: with respect to the variability of the OL in the portfolio 
                                        > Relative: with respect to the variability of the OL in the parent node...
    Returns:
        str: portofolio optimisation layer variability description
    """
    tree_df = splitter.tree_df.copy()
    split_char = '0-S'
    sorted_tree_df = tree_df.sort_values(by='split_gain', ascending=False)
    split_sequence = splitter.split_sequence
    target = 'OL_Ass'
    pf_var_ol = portofolio[target].var()

    if previous_descp!=None:
        descp = previous_descp
        split_node = split_sequence[len(previous_descp.split('|'))]
        child_node_l = splitter.nodes_infos[split_node]['left_child']
        child_node_r = splitter.nodes_infos[split_node]['rigth_child']
        
        parent_var  = portofolio[pf_dict_encoded[split_node]['decision_rule']][target].var()
        lchild_var = portofolio[pf_dict_encoded[child_node_l]['decision_rule']][target].var()
        rchild_var = portofolio[pf_dict_encoded[child_node_r]['decision_rule']][target].var()
        l_prop = pf_dict_encoded[child_node_l]['decision_rule'].sum() / pf_dict_encoded[split_node]['decision_rule'].sum()
        r_prop = pf_dict_encoded[child_node_r]['decision_rule'].sum() / pf_dict_encoded[split_node]['decision_rule'].sum()
        
        if gain_relatif : 
            current_desc = 1 - (  l_prop * lchild_var +  r_prop * rchild_var)/parent_var
        else :
            current_desc = (parent_var - (  l_prop * lchild_var +  r_prop * rchild_var)) / pf_var_ol
        descp = descp + '|' + str(max(round(current_desc, precision),0.0) + 0.0)
        return descp
         
    if nb_desc==-1:
        nb_desc = len(split_sequence)
    else:
        nb_desc = min(nb_desc, len(split_sequence))
    
    # Si aucune description n'est passée on calcul toutes les descriptions jusqu'au nombre de mésures ... 
    descp = ''
    for split_node in split_sequence[:nb_desc]:
        child_node_l = splitter.nodes_infos[split_node]['left_child']
        child_node_r = splitter.nodes_infos[split_node]['rigth_child']
        
        parent_var  = portofolio[pf_dict_encoded[split_node]['decision_rule']][target].var()
        lchild_var = portofolio[pf_dict_encoded[child_node_l]['decision_rule']][target].var()
        rchild_var = portofolio[pf_dict_encoded[child_node_r]['decision_rule']][target].var()
        l_prop = pf_dict_encoded[child_node_l]['decision_rule'].sum() / pf_dict_encoded[split_node]['decision_rule'].sum()
        r_prop = pf_dict_encoded[child_node_r]['decision_rule'].sum() / pf_dict_encoded[split_node]['decision_rule'].sum()      
        if gain_relatif: 
            current_desc = 1 - (  l_prop * lchild_var +  r_prop * rchild_var)/parent_var
        else :
            current_desc = (parent_var - (  l_prop * lchild_var +  r_prop * rchild_var)) / pf_var_ol
        descp = descp + str(max(round(current_desc, precision),0.0) + 0.0) + '|'
    return descp[:-1]


# Some usefull functions to switch from decimal (previous encode) to a binary description of the portfolio ...

def frac2bit_half_precision(frac):
    """half precision encoding 

    Args:
        frac (str): string representing a single measurement on a portofolio

    Returns:
        str: 16 bits (half precision)  representation of frac
    """
    return bin(np.float16(frac).view('H'))[2:].zfill(16)


def precision2bitsize(p):
    """get the number of bits to represent the measurment from the precision order

    Args:
        p (int): precision order of measurment

    Returns:
        int: len on binary representation to output
    """
    return math.floor(np.log2(10**p - 1)) + 1


## binary representation of the decimal part considered as an integer 
def frac2bits(frac, precision = 4, decimal_char = '.'):
    """decimal to bit encoding, using decimal part as integir (Dec2Int)

    Args:
        frac (str): string representing a single measurement on a portofolio
        precision (int, optional): precision order of the measurement. Defaults to 4.
        decimal_char (str, optional): decimal part char identifier Defaults to '.'.

    Returns:
        str: DecAsInt binary representation of frac
    """
    bit_rep = frac[0] 
    dec_part = frac.split(decimal_char, 1)[1][:precision] # Get the precision first digit of the decimal part 
    dec_part = dec_part + '0'*(precision - len(dec_part)) # Complete on the right with 0's if necessary
    dec_part_as_bit = "{0:b}".format(int(dec_part))       # Convert to bits    
    bitstring = '0' * (precision2bitsize(precision) - len(dec_part_as_bit)) + dec_part_as_bit # Add zeros to have a fixed length binary representation 
    return bit_rep + bitstring


### As many 1s as there are policyholders who check the split in a sequence of predefined length
def frac2bits_v2(frac, precision=4):
    """A binary representation is constructed whose length is determined by the precision. In case of: 
        > maximum dispersion (frac = 50%) The heterogeneity of the string is maximum 50 % of 0 and 50 % of 1
        > minimum dispersion (frac = 0 % or frac = 100%) The heterogeneity of the chain is minimal 100 % of 0 or 100 % of 1

    Args:
        frac (str): string representing a single measurement on a portofolio
        precision (int, optional): precision order of the measurement. Defaults to 4.
        decimal_char (str, optional): decimal part char identifier Defaults to '.'.

    Returns:
        str: Simple binary representation of frac
    """

    size = 20 + precision2bitsize(precision)
    nb_ones = round(float(frac)* size)
    bit_rep = '0'* (size- nb_ones) + '1'*nb_ones  
    return bit_rep #bit_rep

def gain_var2bits(gain, precision=4):  
    """adapted version of frac2bits_v2 for OL volatility gain
    Args:
        gain (str): string representing a single OL variability gain measurement on a portofolio
        precision (int, optional): precision order of the measurement. Defaults to 4.

    Returns:
        str: Simple binary representation of frac
    """
    size = precision2bitsize(precision)
    nb_ones = round((float(gain) / 2) * size)
    bit_rep = '0'* (size- nb_ones) + '1'*nb_ones  
    return bit_rep


def decimal_desc2binary_desc(decimal_desc, precision = 4, sep='|', decimal_char = '.', previous_bin_descp=None, previous_dec_descp=None , dec2bin = frac2bits):
    """converts a decimal description sequence into a binary description sequence 

    Args:
        decimal_desc (str): a protofolio description (sequence of measurments) 
        precision (int, optional): precision order of the measurement. Defaults to 4.
        sep (str, optional): measurments separator. Defaults to '|'.
        decimal_char (str, optional): decimal part char identifier Defaults to '.'.
        previous_bin_descp (str, optional): previous description as decimal. Defaults to None.
        previous_dec_descp (str, optional): previous binary representation of description. Defaults to None.
        dec2bin (function, optional): how to get binary representation from a decimal measurment. Defaults to frac2bits.
    Returns:
        str: sequence of binary representation of a protofolio description
    """
    binary_desc = []
    if previous_bin_descp!=None:
        decimal_desc = decimal_desc.replace(previous_dec_descp + '|', '', 1)
        binary_desc = binary_desc + previous_bin_descp       
    desc_as_list = np.array(decimal_desc.split("|")) 
    binary_desc = binary_desc + [ dec2bin(item, precision, decimal_char) for item in desc_as_list]   
    return binary_desc


# Complexity calculation with pyBDM
def Kolmogorov_Complexity_BDM(description_freq, description_ol, normalized=False, trajectory=True, block_len=4, precision=4,frac2bits=frac2bits, gain_var2bits=frac2bits, bdm_type='ignore'):
    """estimate Kolmogorov Complexity of the description using BDM method

    Args:
        description_freq (str): sequence of portofolio description by frequence
        description_ol (str): sequence of  portofolio description by OL variability split gain
        normalized (bool, optional): set to true if Kolmogorov complexity should be normalized. Defaults to False.
        trajectory (bool, optional): if return all complexity fot the increasing description set to True, otherwise return only the last value. Defaults to True.
        block_len (int, optional): block decomposition length to perform in the BDM decomposition. Defaults to 4.
        precision (int, optional): measurments precisions. Defaults to 4.
        frac2bits (function, optional): how to get binary representation from frequence description. Defaults to frac2bits.
        gain_var2bits (function, optional): how to get binary representation from OL variability split gain description. Defaults to frac2bits.
        bdm_type (str, optional): BDM partition type type. Defaults to 'ignore'.

    Returns:
        (np.array, np.array) : sequences of kolmogorov complexity for increasing description of the portofolio
    """
    split_measure = '|'
    desc_freq_as_list = description_freq.split(split_measure)
    desc_ol_as_list = description_ol.split(split_measure)
    nb_measure = len(desc_ol_as_list)
    if bdm_type=='ignore':
        bdm_freq = BDM(ndim=1, shape=(block_len,))
        bdm_ol = BDM(ndim=1, shape=(block_len,))
    elif bdm_type=='PartitionCorrelated':
        bdm_freq = BDM(ndim=1, shape=(block_len,), partition=PartitionCorrelated, shift=1)
        bdm_ol = BDM(ndim=1, shape=(block_len,), partition=PartitionCorrelated, shift=1)
    elif bdm_type=='PartitionRecursive':
        bdm_freq = BDM(ndim=1, shape=(block_len,), partition=PartitionRecursive, min_length=1)
        bdm_ol = BDM(ndim=1, shape=(block_len,), partition=PartitionRecursive, min_length=1)  
    
    
    if trajectory:
        k_bdm_freq = np.zeros(nb_measure)
        k_bdm_ol = np.zeros(nb_measure)
        k_bdm_freq_ol = np.zeros(nb_measure)
        
        for d in range(1,nb_measure+1):
            pf_freq_desc_dec = '|'.join(desc_freq_as_list[:d])
            pf_OL_desc_dec = '|'.join(desc_ol_as_list[:d])
            pf_freq_desc_bin = decimal_desc2binary_desc(pf_freq_desc_dec, precision=precision, dec2bin=frac2bits)
            pf_OL_desc_bin = decimal_desc2binary_desc(pf_OL_desc_dec, precision=precision, dec2bin = gain_var2bits) #gain_var2bits
            
            binary_encoding_freq = np.array([int(bit) for bit in "".join(pf_freq_desc_bin)])
            binary_encoding_OL = np.array([int(bit) for bit in "".join(pf_OL_desc_bin)])    
            i = d-1
            k_bdm_freq[i] = bdm_freq.bdm(binary_encoding_freq, normalized=normalized) 
            k_bdm_ol[i] = bdm_ol.bdm(binary_encoding_OL, normalized=normalized) 
    else:
        pf_freq_desc_bin = decimal_desc2binary_desc(description_freq, precision=precision, dec2bin=frac2bits)
        pf_OL_desc_bin = decimal_desc2binary_desc(description_ol, precision=precision, dec2bin=gain_var2bits)

        # Prepare binary string for BDM ... 
        binary_encoding_freq = np.array([int(bit) for bit in "".join(pf_freq_desc_bin)])
        binary_encoding_OL = np.array([int(bit) for bit in "".join(pf_OL_desc_bin)])    
        
        k_bdm_freq = bdm_freq.bdm(binary_encoding_freq, normalized=normalized) 
        k_bdm_ol = bdm_ol.bdm(binary_encoding_OL, normalized=normalized) 
    return (k_bdm_freq, k_bdm_ol)

# usefull function for Kolmogorov_Complexity_Entropy 
# to generate maximale dispertion sequence (relative vision) starting like : 50%|25%|12.5%| .... 
def make_desc_freq_pf_unif(order, precision, splitter):
    """generates maximale dispertion description  

    Args:
        order (int): description length to generate
        precision (int, optional): measurments precisions
        splitter (LgbmBasedSplitterTree): trained splitter

    Returns:
        str: portofolio frequence description
    """
    split_seq = splitter.split_sequence
    desc = ''
    for node in split_seq[:order]:
        node_d = splitter.tree_df.loc[node, 'node_depth']
        desc = desc + str(round(1/(2**(node_d)), precision)) + '|'
    return desc[:-1]

# usefull function for Kolmogorov_Complexity_Entropy
def single_entropy(prop):
    if prop*(1-prop)==0:
        return 0.0
    else:
        return -prop*np.log2(prop) - (1-prop)*np.log2(1-prop)

def Kolmogorov_Complexity_Entropy(description_freq, description_ol, prop_abs, trajectory=True, precision=4, normalized=False, N_Ass=100_000, splitter=None):
    """estimate Kolmogorov Complexity of the portofolio description based on entropy as describe in the final thesis

    Args:
        description_freq (str): sequence of portofolio description by frequence
        description_ol (str): sequence of  portofolio description by OL variability split gain
        trajectory (bool, optional): if return all complexity fot the increasing description set to True, otherwise return only the last value. Defaults to True.
        precision (int, optional): measurments precisions. Defaults to 4.
        prop_abs (str): sequence of frequence measurments in absolute way
        normalized (bool, optional): if true return a normalized version of KC. Defaults to False.
        N_Ass (int, optional): market size. Defaults to 100_000.
        splitter (LgbmBasedSplitterTree): trained splitter. Defaults to None.

    """
    split_measure = '|'
    desc_freq_as_list = np.array([float(frac) for frac in description_freq.split(split_measure)])
    desc_freq_as_list = np.nan_to_num(desc_freq_as_list)
    desc_ol_as_list = np.array([float(gain) for gain in description_ol.split(split_measure)])
    desc_ol_as_list = np.nan_to_num(desc_ol_as_list)
    
    prop_abs = [float(item) for item in prop_abs.split(split_measure)]
    weight = (prop_abs / desc_freq_as_list)*N_Ass
    weight = np.nan_to_num(weight, posinf=0.0, neginf=0.0)
    nb_measure = len(desc_ol_as_list)
    
    
    K_Entropy_freq = np.zeros(nb_measure)
    K_Entropy_ol = np.zeros(nb_measure)
    K_Entropy_freq[0] = single_entropy(float(desc_freq_as_list[0]))*weight[0]
    K_Entropy_ol[0] = single_entropy(float(desc_ol_as_list[0]))*weight[0]
    if normalized == False:
        for d in range(1, nb_measure):

            K_Entropy_freq[d] = K_Entropy_freq[d-1] + single_entropy(round(float(desc_freq_as_list[d]), precision)) *weight[d]
            K_Entropy_ol[d] = K_Entropy_ol[d-1] + single_entropy(round(float(desc_ol_as_list[d]), precision))*weight[d]
    else:
        unif_pf_desc = make_desc_freq_pf_unif(nb_measure, precision, splitter)
        unif_pf_desc_abs_as_list = np.array([float(frac) for frac in unif_pf_desc.split(split_measure)])
        unif_pf_desc_relative_as_list = np.array([0.5]*nb_measure)
        weight_unif = unif_pf_desc_abs_as_list / unif_pf_desc_relative_as_list
        k_max = 1
        for d in range(1, nb_measure):
            K_Entropy_freq[d] = K_Entropy_freq[d-1]*k_max + single_entropy(round(float(desc_freq_as_list[d]), precision))*weight[d]
            K_Entropy_ol[d] = K_Entropy_ol[d-1]*d + single_entropy(round(float(desc_ol_as_list[d]), precision))*weight[d]
            
            k_max = k_max + round(weight[d], precision)
            
            K_Entropy_freq[d] = K_Entropy_freq[d] / (k_max)
            K_Entropy_ol[d] = K_Entropy_ol[d] / (d + 1)
                
    if trajectory:
        return (K_Entropy_freq, K_Entropy_ol)
    else:
        return (K_Entropy_freq[-1], K_Entropy_ol[-1])

# Approximation of unobserved mutualization based on Shannon Entropy 


def Estimate_Unobserved_Demut(pf_dict_encoded, splitter, nb_desc=-1, trajectory=True, normalized=False, proba_marche=True, verbose=False):
    """Estimates unobserved mutualization for a fixed horizon of portfolio description ... 

    Args:
        pf_dict_encoded (dict): output of the function encode_portofolio_as_dict, portofio by node splitter representation
        splitter (LgbmBasedSplitterTree): trained splitter.
        nb_desc (int, optional): number of measures to be performed on the portfolio. 
                                 Defaults to -1 meaning untill the end of the splitter.
        trajectory (bool, optional): Return the sequence of unobserved mutualisation measurements fot the increasing description set to True, 
                                    otherwise return only the last value. Defaults to True.
        normalized (bool, optional): if the output should be normalized between 0 and 1. Defaults to False.
        proba_marche (bool, optional): if the maximal dispersion should the market one (given by the trained splitter) or uniform. Defaults to True.
        verbose (bool, optional): Defaults to False.

    Returns:
        (np.array, np.array): sequences of unobserved mutualisation estimated
    """
    #Nomnre/Rang de la mésure ...
    nb_measure_max = splitter.nodes_infos['0-S0']['terminal_leaves_nb'] - 1 # Node alphe 0-S0
    if nb_desc != -1:
        nb_measure_max = min(nb_desc, nb_measure_max)
    if trajectory:
        shannon_entropy_prob = np.zeros(nb_measure_max)
        shannon_entropy_unif = np.zeros(nb_measure_max)
        previous_terminal_node = set()
        entropy_measure_prop = 0
        entropy_measure_unif = 0
        mod_print = nb_measure_max // 10
        for d in range(0, nb_measure_max + 1):
            print("Itération", d) if (d%mod_print == 0 ) & verbose else None
            current_terminal_nodes = splitter.current_measure_teriminal_nodes[d]
            
            droped_nodes = set(previous_terminal_node) - set(current_terminal_nodes)
            new_nodes = set(current_terminal_nodes) - set(previous_terminal_node)
            
            for old_node in droped_nodes:
                nb_assure = pf_dict_encoded[old_node]['decision_rule'].sum()  
                entropy_measure_prop = entropy_measure_prop - nb_assure * splitter.nodes_infos[old_node]['unit_entropy']
                entropy_measure_unif = entropy_measure_unif - nb_assure * np.log2(splitter.nodes_infos[old_node]['terminal_leaves_nb'])
            for new_node in new_nodes:
                nb_assure = pf_dict_encoded[new_node]['decision_rule'].sum()  
                entropy_measure_prop = entropy_measure_prop + nb_assure * splitter.nodes_infos[new_node]['unit_entropy']
                entropy_measure_unif = entropy_measure_unif + nb_assure * np.log2(splitter.nodes_infos[new_node]['terminal_leaves_nb'])
                
            shannon_entropy_prob[d-1] = entropy_measure_prop
            shannon_entropy_unif[d-1] = entropy_measure_unif
            previous_terminal_node = current_terminal_nodes
            
    else:
            current_terminal_nodes = splitter.current_measure_teriminal_nodes[nb_measure_max]
            entropy_measure_prop = 0
            entropy_measure_unif = 0
            for ter_node in current_terminal_nodes:
                nb_assure = pf_dict_encoded[ter_node]['decision_rule'].sum()        
                if '0-S' in ter_node:
                    entropy_measure_prop = entropy_measure_prop + nb_assure * splitter.nodes_infos[ter_node]['unit_entropy']
                    entropy_measure_unif = entropy_measure_unif + nb_assure * np.log2(splitter.nodes_infos[ter_node]['terminal_leaves_nb'])

            shannon_entropy_prob = entropy_measure_prop
            shannon_entropy_unif = entropy_measure_unif
    n_ass = pf_dict_encoded['0-S0']['decision_rule'].sum()
    entropy_factor_normalizing =  np.log2(splitter.nodes_infos['0-S0'] ['terminal_leaves_nb'])
    if normalized:
        shannon_entropy_prob = shannon_entropy_prob / (n_ass * entropy_factor_normalizing)
        shannon_entropy_unif = shannon_entropy_unif / (n_ass * entropy_factor_normalizing)
    return shannon_entropy_prob, shannon_entropy_unif

# ------------------------
# Mutualisation index base on algorithmic entropie measure
def Entropy_Algo_Demut(splitter, pf_dict_encoded, description_freq, description_ol, prop_abs, trajectory=True, normalized = True, precision=4, proba_marche=True, N_Ass =100_000):
    """mutualisation index base on algorithmic entropie

    Args:
        splitter (LgbmBasedSplitterTree): trained splitter.
        pf_dict_encoded (dict): output of the function encode_portofolio_as_dict, portofio by node splitter representation
        description_freq (str): sequence of portofolio description by frequence
        description_ol (str): sequence of  portofolio description by OL variability split gain
        prop_abs (str): sequence of frequence measurments in absolute way
        trajectory (bool, optional): to return the sequence for index  for the increasing description set to True, 
                                    otherwise return only the last value. Defaults to True.
        normalized (bool, optional): if the index should be normalized between 0 and 1. Defaults to True.
        precision (int, optional): measurments precisions Defaults to 4.
        proba_marche (bool, optional): if the maximal dispersion should the market one (given by the trained splitter) or uniform. Defaults to True.
        verbose (bool, optional): Defaults to False.
        N_Ass (_type_, optional): market size. Defaults to N_Ass.

    Returns:
        (np.array, np.array, np.array): generalized mutualization ratio,
                                        observed mutualisation estimate using Kolmogorov complexity
                                        unobserved mutualisation estimate using Shannon entropy
    """
    split_measure = '|'
    desc_freq_as_list = description_freq.split(split_measure)
    nb_measure = len(desc_freq_as_list)
    entropy_factor_normalizing =  np.log2(splitter.nodes_infos['0-S0'] ['terminal_leaves_nb'])
    
    K_Entropy_Freq, K_Entropy_ol = Kolmogorov_Complexity_Entropy(description_freq = description_freq, 
                                                                  description_ol=description_ol, 
                                                                  prop_abs=prop_abs,
                                                                  normalized=False, 
                                                                  trajectory=trajectory, 
                                                                  precision = precision)


    shannon_entropy_prob, shannon_entropy_unif = Estimate_Unobserved_Demut(pf_dict_encoded=pf_dict_encoded, 
                                                                   splitter=splitter, 
                                                                   nb_desc = nb_measure, 
                                                                   trajectory=trajectory, 
                                                                   normalized = False, 
                                                                   proba_marche = proba_marche)


    shannon_entropy = shannon_entropy_prob if proba_marche else shannon_entropy_unif
    if normalized:
        K_Entropy_Freq = K_Entropy_Freq / (N_Ass * entropy_factor_normalizing)
        K_Entropy_ol = K_Entropy_ol / (N_Ass * entropy_factor_normalizing)
        shannon_entropy =  shannon_entropy / (N_Ass * entropy_factor_normalizing)
    Entropy_Algo_Demut_freq = K_Entropy_Freq + shannon_entropy
    Entropy_Algo_Demut_ol = K_Entropy_ol + shannon_entropy
    
    Ent_Algos = (Entropy_Algo_Demut_freq, Entropy_Algo_Demut_ol)
    K = (K_Entropy_Freq, K_Entropy_ol)
    H = shannon_entropy
    return (Ent_Algos, K, H)