import matplotlib.pyplot as plt
import shap
import yaml
import argparse
import xgboost as xgb 
from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml import plot_utils
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml_converter.h4ml_converter import H4MLConverter

def ML_DMesons ():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgFileName', default='Config_MLDmesonsScript.yml', help='config file name for ml')
    args = parser.parse_args()

    print('Loading analysis configuration: ...', end='\r')
    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    print('Loading analysis configuration: Done!')
    
    hdl_kaka_mc_signal = TreeHandler(inputCfg['input']['mcKaKa'], inputCfg['input']['treename'])
    hdl_kaka_data_bkg = TreeHandler(inputCfg['input']['dataKaKa'], inputCfg['input']['treename'])

    hdl_kaka_mc_signal.apply_preselections(inputCfg['dataPrep']['Signal'])
    hdl_kaka_data_bkg.apply_preselections(inputCfg['dataPrep']['filt_bkg_TagsInvMass'])
    
    hdl_all = [hdl_kaka_data_bkg, hdl_kaka_mc_signal]
    
    vars_to_draw = inputCfg['Output']['plotLabels']
    leg_labels = inputCfg['Output']['legLabels']
    
    plot_utils.plot_distr(hdl_all, vars_to_draw, bins=250, labels=leg_labels, log=False, density=True, figsize=(12, 7), alpha=0.3, grid=False)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
    plt.show()
    plt.savefig(inputCfg['Output']['dir']+'TopoVars.pdf')
    
    plot_utils.plot_corr(hdl_all, vars_to_draw, leg_labels)
    plt.show()
    plt.savefig(inputCfg['Output']['dir']+'TopoVarsCorr.pdf')
    
    ## ML
    train_test_data = train_test_generator(hdl_all, [0, 1], test_size=0.2, random_state=42)
    trainset = train_test_data[0]
    ytrain = train_test_data[1]
    testset = train_test_data[2]
    ytestset = train_test_data[3]
    
    # The model 
    features_for_train = inputCfg['ml']['training_vars']
    model_params = {'max_depth':5, 'learning_rate':0.029, 'n_estimators':500, 'min_child_weight':2.7, 'subsample':0.90, 'colsample_bytree':0.97, 'n_jobs':1, 'tree_method': 'hist'}
    model_clf = xgb.XGBClassifier() 
    model_hdl = ModelHandler(model_clf, features_for_train, model_params)

    # Training and testing the model
    model_hdl.train_test_model(train_test_data, multi_class_opt="ovo")
    y_pred_train = model_hdl.predict(train_test_data[0], False)
    y_pred_test = model_hdl.predict(train_test_data[2], False)
    
    plt.rcParams["figure.figsize"] = (10, 7)
    ml_out_fig = plot_utils.plot_output_train_test(model_hdl, train_test_data, 100, False, leg_labels, True, density=True)
    plt.savefig(inputCfg['Output']['dir']+'TopoVarsTraining.pdf')

    plot_utils.plot_roc(ytestset, y_pred_test, multi_class_opt="ovo")
    plt.savefig(inputCfg['Output']['dir']+'ROC_testSet.pdf')

    plot_utils.plot_roc(ytrain, y_pred_train, multi_class_opt="ovo")
    plt.savefig(inputCfg['Output']['dir']+'ROC_trainSet.pdf')
    
    plot_utils.plot_feature_imp(train_test_data[2], train_test_data[3], model_hdl) 
    plt.savefig(inputCfg['Output']['dir']+'ROC_featureImp.pdf')

    #explainer = shap.Explainer(model_clf)
    #shap_values = explainer(hdl_k)
    #shap.plots.waterfall(shap_values[0])
    #plt.savefig('/home/dmorris/ML-TagAndProbeDmesons/outputs/Shap_ROC_featureImp.pdf')
    
    model_hdl.dump_model_handler(inputCfg['ml']['modelHdl'])
    model_hdl.dump_original_model(inputCfg['ml']['modelOriginal'])
    
    model_converter = H4MLConverter(model_hdl)
    model_onnx = model_converter.convert_model_onnx(1, len(features_for_train))
    model_converter.dump_model_onnx(inputCfg['ml']['modelonnx']) 


    
    
