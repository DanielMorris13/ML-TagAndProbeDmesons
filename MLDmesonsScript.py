import matplotlib.pyplot as plt
import xgboost as xgb # gradient boosting
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml import plot_utils
from hipe4ml.analysis_utils import train_test_generator

def ML_DMesons ():
    hdl_mc = TreeHandler("../tag_and_probe/AO2D_pp_LHC22b1b_run302027_KaKaFromDsOrDplus_Mc_merged.root","O2topvars")
    hdl_mc.apply_preselections("fisSignal = 1")
    
    vars_to_draw = ['fTagsInvMass', 'fDecayLength', 'fDecayLengthXY', 'fDecayLengthNormalised', 'fTrackDcaXY', 'fCpa', 'fCpaXY']
    leg_label = ["Mc Signal"]
    
    plot_utils.plot_distr(hdl_mc, vars_to_draw, bins=1000, labels=leg_labels, log=True, density=True, figsize=(12, 7), alpha=0.3, grid=False)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
    plt.show()
    
