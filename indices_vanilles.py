"""Une fonction pour évaluer les indicateurs simples de mutualisation 
   sur un portefeuille assurantiel
"""

import pandas as pd
import numpy as np

def Ent(vect, round_order=2):
    """Compute entropy for a geven vector vect

    Args:
        vect (np.array): float vector, representing burning cost or optimisation layer
        round_order (int, optional): order to round the vector vect,  defaults to 2.

    Returns:
        float: entropy of the given vector
    """
    summary = pd.Series(vect.round(round_order)).value_counts(normalize=True).sort_index()
    unique_values = summary.index
    freqs = summary.values
    return - np.dot(freqs, np.log2(freqs))

def Indices_vanilles(BC_Omega, BC_Ass, OL_Ass, round_order=2):
    """ Calculates all vanilla indices based on risk profile,
        pure premium and optimization layer

    Args:
        BC_Omega (np.array): the most individualized risk profiles of policyholders, calculated from a hypersegmented model
        BC_Ass (np.array): the technical premium as implemented in the insurer's commercial grid
        OL_Ass (np.array): optimisation layer strategy of the insurer
        round_order (int, optional): order to round the premium vectors,  defaults to 2.

    Returns:
        dict : dictionary of simple indicators
    """
    # len of premium informations should be the same
    assert (len(BC_Ass) == len(BC_Omega)) & (len(OL_Ass) == len(BC_Omega))

    ID_Var_BC = np.var(BC_Ass.round(round_order)) / np.var(BC_Omega)

    ID_Ent_BC = Ent(BC_Ass, round_order=round_order) / np.log2(len(BC_Ass))

    IM_Var_OL = np.var(OL_Ass.round(round_order)) / np.var(BC_Omega)

    IM_Ent_OL = Ent(OL_Ass, round_order=round_order) / np.log2(len(OL_Ass))

    IM_L1_BC = np.mean(np.abs(y_true=BC_Omega, y_pred=BC_Ass.round(round_order)))

    # Compute expected result, it should be close to zero
    ER = np.mean(BC_Ass - BC_Omega)
    print(f"Expected Result :{ER}")

    IM_L1_BC_Shift = IM_L1_BC - np.abs(ER)

    r = {
        'ID_Ent_BC': ID_Ent_BC,
        'IM_L1_BC': IM_L1_BC,
        'ID_Var_BC': ID_Var_BC,
        'IM_Var_OL': IM_Var_OL,
        'IM_Ent_OL': IM_Ent_OL,
        'IM_L1_BC_Shift': IM_L1_BC_Shift,
    }

    return r
