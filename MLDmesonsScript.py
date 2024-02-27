import matplotlib.pyplot as plt
import shap
import xgboost as xgb # gradient boosting
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml import plot_utils
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml_converter.h4ml_converter import H4MLConverter

def ML_DMesons ():
    hdl_kaka_mc_signal = TreeHandler("/home/dmorris/Data/AO2D_pp_LHC22b1b_run302027_KaKaFromDsOrDplus_Mc_merged.root","O2topovars")
    #hdl_kaka_mc_bkg = TreeHandler("/home/dmorris/Data/AO2D_pp_LHC22b1b_run302027_KaKaFromDsOrDplus_Mc_merged.root","O2topovars")

    hdl_kaka_data_bkg = TreeHandler("/home/dmorris/Data/AO2D_pp_DATA_LHC22q_pass4_lowIR_run528991_KaKaFromDsOrDplus_Data_merged.root","O2topovars")

    hdl_kaka_mc_signal.apply_preselections("fIsSignal == 1")
    #hdl_kaka_mc_bkg.apply_preselections("fIsSignal == 0")
    hdl_kaka_data_bkg.apply_preselections("fTagsInvMass < 1 or fTagsInvMass > 1.05")
    
    hdl_all = [hdl_kaka_data_bkg, hdl_kaka_mc_signal]
    
    vars_to_draw = ['fTagsInvMass', 'fDecayLength', 'fDecayLengthXY', 'fDecayLengthNormalised', 'fTrackDcaXY', 'fCpa', 'fCpaXY']
    leg_labels = ["Data Bkg", "Mc Signal",]
    
    plot_utils.plot_distr(hdl_all, vars_to_draw, bins=250, labels=leg_labels, log=False, density=True, figsize=(12, 7), alpha=0.3, grid=False)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
    plt.show()
    plt.savefig('/home/dmorris/ML-TagAndProbeDmesons/outputs/TopoVars.pdf')
    
    plot_utils.plot_corr(hdl_all, vars_to_draw, leg_labels)
    plt.show()
    plt.savefig('/home/dmorris/ML-TagAndProbeDmesons/outputs/TopoVarsCorr.pdf')
    
    ## ML
    train_test_data = train_test_generator(hdl_all, [0, 1], test_size=0.2, random_state=42)
    trainset = train_test_data[0]
    ytrain = train_test_data[1]
    testset = train_test_data[2]
    ytestset = train_test_data[3]
    
    # The model 
    features_for_train = ['fDecayLength', 'fDecayLengthXY', 'fDecayLengthNormalised', 'fTrackDcaXY', 'fCpa', 'fCpaXY']
    model_params = {'max_depth':5, 'learning_rate':0.029, 'n_estimators':500, 'min_child_weight':2.7, 'subsample':0.90, 'colsample_bytree':0.97, 'n_jobs':1, 'tree_method': 'hist'}
    model_clf = xgb.XGBClassifier() 
    model_hdl = ModelHandler(model_clf, features_for_train, model_params)

    # Training and testing the model
    model_hdl.train_test_model(train_test_data, multi_class_opt="ovo")
    y_pred_train = model_hdl.predict(train_test_data[0], False)
    y_pred_test = model_hdl.predict(train_test_data[2], False)
    
    plt.rcParams["figure.figsize"] = (10, 7)
    ml_out_fig = plot_utils.plot_output_train_test(model_hdl, train_test_data, 100, False, leg_labels, True, density=True)
    plt.savefig('/home/dmorris/ML-TagAndProbeDmesons/outputs/TopoVarsTraining.pdf')

    plot_utils.plot_roc(ytestset, y_pred_test, multi_class_opt="ovo")
    plt.savefig('/home/dmorris/ML-TagAndProbeDmesons/outputs/ROC_testSet.pdf')

    plot_utils.plot_roc(ytrain, y_pred_train, multi_class_opt="ovo")
    plt.savefig('/home/dmorris/ML-TagAndProbeDmesons/outputs/ROC_trainSet.pdf')
    
    plot_utils.plot_feature_imp(train_test_data[2], train_test_data[3], model_hdl) 
    plt.savefig('/home/dmorris/ML-TagAndProbeDmesons/outputs/ROC_featureImp.pdf')

    #explainer = shap.Explainer(model_clf)
    #shap_values = explainer(hdl_k)
    #shap.plots.waterfall(shap_values[0])
    #plt.savefig('/home/dmorris/ML-TagAndProbeDmesons/outputs/Shap_ROC_featureImp.pdf')
    
    model_hdl.dump_model_handler("/home/dmorris/ML-TagAndProbeDmesons/outputs/ModelHandler_model.pickle")
    model_hdl.dump_original_model("/home/dmorris/ML-TagAndProbeDmesons/outputs/XGBoos_model.pickle")
    
    model_converter = H4MLConverter(model_hdl) # create the converter object
    model_onnx = model_converter.convert_model_onnx(1, len(features_for_train))
    model_converter.dump_model_onnx("/home/dmorris/ML-TagAndProbeDmesons/outputs/model_onnx.onnx") # dump the model in ONNX format  


    
    
