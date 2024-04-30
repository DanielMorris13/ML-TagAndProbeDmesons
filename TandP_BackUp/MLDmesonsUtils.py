#Classe che implementa due principali funzioni Data_split e ML_Dmesons. La prima splitta i dataset mc e dati in training/test e app.
#la seconda funzione applica ML e ottimizzazione degli iperparametri, e salva gli score sia per dati che mc app. 
import ROOT
import matplotlib.pyplot as plt
import shap
import yaml
import argparse
import pickle
import xgboost as xgb 
import optuna as op
import plotly
import numpy as np
import array
import pandas as pd

from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml import plot_utils
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml_converter.h4ml_converter import H4MLConverter

def Data_split():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgFileName', default='Config_MLDmesonsScript.yml', help='config file name for ml')
    args = parser.parse_args()

    print('Loading analysis configuration: ...', end='\r')
    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    print('Loading analysis configuration: Done!')
    
    hdl_mc = TreeHandler(inputCfg['input']['mcKaKa'], inputCfg['input']['treename'])
    hdl_data = TreeHandler(inputCfg['input']['dataKaKa'], inputCfg['input']['treename'])
    
    print(hdl_mc)
    print(hdl_data)

    df_mc = hdl_mc.get_data_frame()
    df_data = hdl_data.get_data_frame()
    
    df_mc_ml = df_mc.iloc[:int(len(df_mc)*0.8), :]
    df_mc_app = df_mc.iloc[int(len(df_mc)*0.8):, :]
    df_data_ml = df_data.iloc[:int(len(df_data)*inputCfg['dataPrep']['DFfraction_data_ml']), :]
    df_data_app = df_data.iloc[int(len(df_data)*inputCfg['dataPrep']['DFfraction_data_ml']):, :]

    print(len(df_mc_ml))
    print(len(df_mc_app))
    print(len(df_data_ml))
    
    df_mc_ml.to_parquet(inputCfg['Output']['dir']+'kaka_mc_dataset_ml_80.gzip', compression='gzip', index=False)
    df_mc_app.to_parquet(inputCfg['Output']['dir']+'kaka_mc_dataset_app_20.gzip', compression='gzip', index=False)
    #df_data_ml.to_parquet(inputCfg['Output']['dir']+'kaka_data_dataset_ml.gzip', compression='gzip', index=False)
    #df_data_app.to_parquet(inputCfg['Output']['dir']+'kaka_data_dataset_app.gzip', compression='gzip', index=False)

def ML_DMesons (PTmin, PTmax):
    
    #read the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgFileName', default='Config_MLDmesonsScript.yml', help='config file name for ml')
    args = parser.parse_args()

    print('Loading analysis configuration: ...', end='\r')
    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    print('Loading analysis configuration: Done!')
    
    #Get the datasets in paquet format - and the same size datasaet for data and mc
    hdl_mc_ml = TreeHandler(inputCfg['Output']['dir']+'kaka_mc_dataset_ml.gzip')
    hdl_mc_app = TreeHandler(inputCfg['Output']['dir']+'kaka_mc_dataset_app.gzip')
    
    hdl_data_ml = TreeHandler(inputCfg['Output']['dir']+'kaka_data_dataset_ml.gzip')
    hdl_data_app = TreeHandler(inputCfg['Output']['dir']+'kaka_data_dataset_app.gzip')
    
    print("MC before selections:", len(hdl_mc_ml))
    print("Data before selections:", len(hdl_data_ml))
    
    #Apply preselection on the trainign Set
    hdl_mc_ml.apply_preselections(inputCfg['dataPrep']['Signal'])
    hdl_data_ml.apply_preselections(inputCfg['dataPrep']['filt_bkg_TagsInvMass'])
    
    hdl_data_ml.apply_preselections(f'{PTmin} < fTagsPt < {PTmax}')
    hdl_mc_ml.apply_preselections(f'{PTmin} < fTagsPt < {PTmax}')
    
    #Apply preselection on the Application dataSet in PT bins
    hdl_data_app.apply_preselections(f'{PTmin} < fTagsPt < {PTmax}')
    hdl_mc_app.apply_preselections(f'{PTmin} < fTagsPt < {PTmax}')
    
    print("MC After selections:", len(hdl_mc_ml))
    print("Data After selections:", len(hdl_data_ml))

    hdl_data_ml_reduced = hdl_data_ml.get_subset(size=(len(hdl_mc_ml)), rndm_state = 13)
    
    print("Using same size for the background", len(hdl_data_ml_reduced))
    
    hdl_all = [hdl_data_ml_reduced, hdl_mc_ml]
    
    vars_to_draw = inputCfg['Output']['plotLabels']
    leg_labels = inputCfg['Output']['legLabels']

    plt.clf()
    plot_utils.plot_distr(hdl_all, vars_to_draw, bins=100, labels=leg_labels, log=True, density=True, figsize=(12, 7), alpha=0.3, grid=False)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'TopoVars.pdf')
    
    plt.clf()
    plot_utils.plot_corr(hdl_all, vars_to_draw, leg_labels)
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'TopoVarsCorr.pdf')
    
    ## ML
    train_test_data = train_test_generator(hdl_all, inputCfg['ml']['indexClass'], test_size=0.2, random_state=42)
    trainset = train_test_data[0]
    ytrain = train_test_data[1]
    testset = train_test_data[2]
    ytestset = train_test_data[3]
    
    # The model - Previous to the optimization
    features_for_train = inputCfg['ml']['training_vars']
    model_clf = xgb.XGBClassifier() 
    hyper_pars_pre_opt = {'max_depth':5, 'learning_rate':0.029, 'n_estimators':500, 'min_child_weight':2.7, 'subsample':0.90, 'colsample_bytree':0.97, 'n_jobs':1}
    model_hdl = ModelHandler(model_clf, features_for_train, hyper_pars_pre_opt)
        
    # Training and testing the model
    model_hdl.train_test_model(train_test_data, multi_class_opt="ovo")
    y_pred_train = model_hdl.predict(train_test_data[0], False)
    y_pred_test = model_hdl.predict(train_test_data[2], False)
    
    plt.clf()
    plt.rcParams["figure.figsize"] = (10, 7)
    ml_out_fig = plot_utils.plot_output_train_test(model_hdl, train_test_data, 100, False, leg_labels, True, density=True)
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'TopoVarsTrainingCurve.pdf')

    plt.clf()
    plot_utils.plot_roc(ytestset, y_pred_test, multi_class_opt="ovo")
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'ROC_testSet.pdf')
    
    plt.clf()
    plot_utils.plot_roc(ytrain, y_pred_train, multi_class_opt="ovo")
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'ROC_trainSet.pdf')
    
    #Optimization of the model hyparameters

    plt.clf()
    hyper_pars_ranges = {'max_depth': (2, 10), 'learning_rate':(0.01, 0.1), 'n_estimators':(200, 1500), 'min_child_weight':(2, 10), 'colsample_bytree':(0.1, 1.), 'n_jobs':(1,5)}
    model_hdl.optimize_params_optuna(train_test_data, hyper_pars_ranges, cross_val_scoring="roc_auc_ovo", n_trials=10, direction="maximize", save_study=inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'optuna_study.pkl')
    opstudy = pickle.load(open(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+"optuna_study.pkl", "rb")) # load the study
    plot_opstudy = op.visualization.plot_contour(opstudy)
    plot_opstudy.show()
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'Optuna_parametersImp.pdf')
    plt.clf()
    plot_hypecorr = op.visualization.plot_parallel_coordinate(opstudy) # plot the correlation between the hyperparameters and the relative performance
    plot_hypecorr.show()
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'Optuna_hyperCorr.pdf')
    
    # Training and testing the model - Post optimization
    model_hdl.set_model_params(opstudy.best_params) #feed the model with the best paramteres obtained with optuna
    model_hdl.train_test_model(train_test_data, multi_class_opt="ovo")
    y_pred_train = model_hdl.predict(train_test_data[0], False)
    y_pred_test = model_hdl.predict(train_test_data[2], False)
    
    plt.clf()
    plt.rcParams["figure.figsize"] = (10, 7)
    ml_out_fig = plot_utils.plot_output_train_test(model_hdl, train_test_data, 100, False, leg_labels, True, density=True)
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'TopoVarsTrainingCurve_optimizated.pdf')

    plt.clf()
    plot_utils.plot_roc(ytestset, y_pred_test, multi_class_opt="ovo")
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'ROC_testSet_optimizated.pdf')
    
    plt.clf()
    plot_utils.plot_roc(ytrain, y_pred_train, multi_class_opt="ovo")
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'ROC_trainSet_optimizated.pdf')
    
    #Saving the model
    plt.clf()
    plots_shap = plot_utils.plot_feature_imp(train_test_data[2], train_test_data[3], model_hdl, inputCfg['Output']['legLabels'])
    for i in range(0,len(plots_shap)):
        plots_shap[i].savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'featureImp_'+str(i)+'.pdf')
        
    model_hdl.dump_model_handler(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+"ModelHandler_MulticlassDs_pT_2_4.pickle")
    model_hdl.dump_original_model(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+"XGBoostModel_MulticlassDs_pT_2_4.pickle")
    print("starting: Scores - data")
    
    #Apply the model to the data dataSet - Data
    plt.clf()
    yPred_data = model_hdl.predict(hdl_data_app, False)
    df_data_app = hdl_data_app.get_data_frame()
    df_data_app['score'] = yPred_data
    df_data_app['score'].plot(kind='hist', bins=100, alpha=0.8, log=True, figsize=(12, 7), grid=False, density=True, label='score')
    plt.legend()
    plt.show()
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'data_prediction.pdf')
    df_data_app_toSave = df_data_app[['fTagsInvMass', 'fDecayLength', 'fDecayLengthXY', 'fDecayLengthNormalised', 'fTrackDcaXY', 'fCpa', 'fCpaXY', 'score']].copy()
    print("completed score calc.")
    
    #Apply the model to the data dataSet - App
    plt.clf()
    print("starting mc scores")
    yPred_mc = model_hdl.predict(hdl_mc_app, False)
    df_mc_app = hdl_mc_app.get_data_frame()
    df_mc_app['score'] = yPred_mc
    df_mc_app['score'].plot(kind='hist', bins=100, alpha=0.8, log=True, figsize=(12, 7), grid=False, density=True, label='score')
    plt.legend()
    plt.draw()
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'mc_prediction.pdf')
    df_mc_app_toSave = df_mc_app[['fTagsInvMass', 'fDecayLength', 'fDecayLengthXY', 'fDecayLengthNormalised', 'fTrackDcaXY', 'fCpa', 'fCpaXY', 'fIsSignal', 'score']].copy()
 
    #saving the score-dfs to parquet file .gzip
    print("saving the scores")
    df_mc_app_toSave.to_parquet(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'kaka_mc_dfScores.gzip', compression='gzip', index=False)
    df_data_app_toSave.to_parquet(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}/'+'kaka_data_dfScores.gzip', compression='gzip', index=False)

def purity_and_efficiency(Pcut = 0.5):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgFileName', default='Config_MLDmesonsScript.yml', help='config file name for ml')
    args = parser.parse_args()

    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    
    hdl_data_scores = TreeHandler(inputCfg['Output']['dir']+'mlData/'+'kaka_data_dfScores.gzip')
    hdl_mc_scores = TreeHandler(inputCfg['Output']['dir']+'mlData/'+'kaka_mc_dfScores.gzip')
    
    #to run once
    plt.clf()
    bkg = hdl_mc_scores.apply_preselections("fIsSignal==0", False)
    signals = hdl_mc_scores.apply_preselections("fIsSignal==1", False)
    df_signals = signals.get_data_frame()
    df_bkg = bkg.get_data_frame()
    df_all = hdl_mc_scores.get_data_frame()

    plt.hist(df_signals['score'],100,label='signal')
    plt.hist(df_bkg['score'],100,label='bkg')
    plt.savefig(inputCfg['Output']['dir']+'score_mc_classes.pdf')
    plt.clf()

    all_over_thr = len(df_all.query(f'score > {Pcut}'))
    eff_den = len(df_signals)
    sign_over_thr = df_signals.query(f'score > {Pcut}')
    bkg_over_thr = df_bkg.query(f'score > {Pcut}')
    eff_num = len(sign_over_thr)
    
    purity_den = len(bkg_over_thr)
    print
    print(purity_den)
    purity_num = len(sign_over_thr)
    eff = eff_num / eff_den
    purity = purity_num / all_over_thr
    print(eff, purity)
     
    return purity, eff

def purity_vs_efficiency():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgFileName', default='Config_MLDmesonsScript.yml', help='config file name for ml')
    args = parser.parse_args()

    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    
    Pcuts = [i*0.001 for i in range(1000)]
    
    purity_arr = np.array([])
    eps_arr = np.array([])

    for cut in Pcuts:
        purity, eps = purity_and_efficiency(cut)
        purity_arr = np.append(purity_arr, purity)
        eps_arr = np.append(eps_arr, eps)
        print(purity, eps)
        
    print(purity_arr, eps_arr)
    plt.scatter(eps_arr, purity_arr, color = "red", marker = "*" , s = 20, label = 'data')
    plt.legend(fontsize='x-small')
    plt.ylabel( 'purity' )
    plt.xlabel( 'eps' )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.savefig(inputCfg['Output']['dir']+'purity_vs_eps.pdf')
    

def ML_DMesons_test (PTmin, PTmax):
    
    #read the config file
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgFileName', default='Config_MLDmesonsScript.yml', help='config file name for ml')
    args = parser.parse_args()

    print('Loading analysis configuration: ...', end='\r')
    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    print('Loading analysis configuration: Done!')
    
    #Get the datasets in paquet format - and the same size datasaet for data and mc
    hdl_mc_ml = TreeHandler(inputCfg['Output']['dir']+'kaka_mc_dataset_ml_80.gzip')
    hdl_mc_app = TreeHandler(inputCfg['Output']['dir']+'kaka_mc_dataset_app_20.gzip')
    
    hdl_data_ml = TreeHandler(inputCfg['Output']['dir']+'kaka_data_dataset_ml.gzip')
    hdl_data_app = TreeHandler(inputCfg['Output']['dir']+'kaka_data_dataset_app.gzip')
    
    print("MC before selections:", len(hdl_mc_ml))
    #print("Data before selections:", len(hdl_data_ml))
    
    #Apply preselection on the trainign Set
    hdl_mc_ml.apply_preselections(f'{PTmin} < fTagsPt < {PTmax}')
    
    hdl_mc_ml_signal = hdl_mc_ml.apply_preselections(inputCfg['dataPrep']['Signal'], inplace = False)
    hdl_mc_ml_bkg = hdl_mc_ml.apply_preselections('fIsSignal == 0', inplace = False)
    
    #Apply preselection on the Application dataSet in PT bins
    hdl_data_app.apply_preselections(f'{PTmin} < fTagsPt < {PTmax}')
    hdl_mc_app.apply_preselections(f'{PTmin} < fTagsPt < {PTmax}')
    
    print("MC signal After selections:", len(hdl_mc_ml_signal))
    print("MC bkg After selections:", len(hdl_mc_ml_bkg))

    #hdl_data_ml_reduced = hdl_data_ml.get_subset(size=(len(hdl_mc_ml)), rndm_state = 13)
    
    #print("Using same size for the background", len(hdl_data_ml_reduced))
    
    hdl_all = [hdl_mc_ml_bkg, hdl_mc_ml_signal]
    
    vars_to_draw = inputCfg['Output']['plotLabels']
    leg_labels = inputCfg['Output']['legLabels']

    plt.clf()
    plot_utils.plot_distr(hdl_all, vars_to_draw, bins=100, labels=leg_labels, log=True, density=True, figsize=(12, 7), alpha=0.3, grid=False)
    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.99, top=0.96, hspace=0.55, wspace=0.55)
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'TopoVars.pdf')
    
    plt.clf()
    plot_utils.plot_corr(hdl_all, vars_to_draw, leg_labels)
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'TopoVarsCorr.pdf')
    
    ## ML
    train_test_data = train_test_generator(hdl_all, inputCfg['ml']['indexClass'], test_size=0.2, random_state=42)
    trainset = train_test_data[0]
    ytrain = train_test_data[1]
    testset = train_test_data[2]
    ytestset = train_test_data[3]
    
    # The model - Previous to the optimization
    features_for_train = inputCfg['ml']['training_vars']
    model_clf = xgb.XGBClassifier() 
    hyper_pars_pre_opt = {'max_depth':5, 'learning_rate':0.029, 'n_estimators':500, 'min_child_weight':2.7, 'subsample':0.90, 'colsample_bytree':0.97, 'n_jobs':1}
    model_hdl = ModelHandler(model_clf, features_for_train, hyper_pars_pre_opt)
        
    # Training and testing the model
    model_hdl.train_test_model(train_test_data, multi_class_opt="ovo")
    y_pred_train = model_hdl.predict(train_test_data[0], False)
    y_pred_test = model_hdl.predict(train_test_data[2], False)
    
    plt.clf()
    plt.rcParams["figure.figsize"] = (10, 7)
    ml_out_fig = plot_utils.plot_output_train_test(model_hdl, train_test_data, 100, False, leg_labels, True, density=True)
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'TopoVarsTrainingCurve.pdf')

    plt.clf()
    plot_utils.plot_roc(ytestset, y_pred_test, multi_class_opt="ovo")
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'ROC_testSet.pdf')
    
    plt.clf()
    plot_utils.plot_roc(ytrain, y_pred_train, multi_class_opt="ovo")
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'ROC_trainSet.pdf')
    
    #Optimization of the model hyparameters
    '''
    plt.clf()
    hyper_pars_ranges = {'max_depth': (2, 10), 'learning_rate':(0.01, 0.1), 'n_estimators':(200, 1500), 'min_child_weight':(2, 10), 'colsample_bytree':(0.1, 1.), 'n_jobs':(1,5)}
    model_hdl.optimize_params_optuna(train_test_data, hyper_pars_ranges, cross_val_scoring="roc_auc_ovo", n_trials=10, direction="maximize", save_study=inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'optuna_study.pkl')
    opstudy = pickle.load(open(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+"optuna_study.pkl", "rb")) # load the study
    plot_opstudy = op.visualization.plot_contour(opstudy)
    plot_opstudy.show()
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'Optuna_parametersImp.pdf')
    plt.clf()
    plot_hypecorr = op.visualization.plot_parallel_coordinate(opstudy) # plot the correlation between the hyperparameters and the relative performance
    plot_hypecorr.show()
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'Optuna_hyperCorr.pdf')
    '''

    # Training and testing the model - Post optimization
    #pt 2_4
    #hyper_pars_opt = {'max_depth':3, 'learning_rate':0.0428043003342268, 'n_estimators':769, 'min_child_weight':7, 'subsample':0.90, 'colsample_bytree':0.7343776051880557, 'n_jobs':3}
    #pt 6_10
    hyper_pars_opt = {'max_depth':9, 'learning_rate':0.02781844033317222, 'n_estimators':587, 'min_child_weight':7, 'subsample':0.90, 'colsample_bytree':0.7343776051880557, 'n_jobs':2}
    model_hdl.set_model_params(hyper_pars_opt) #feed the model with the best paramteres obtained with optuna
    model_hdl.train_test_model(train_test_data, multi_class_opt="ovo")
    y_pred_train = model_hdl.predict(train_test_data[0], False)
    y_pred_test = model_hdl.predict(train_test_data[2], False)
    
    plt.clf()
    plt.rcParams["figure.figsize"] = (10, 7)
    ml_out_fig = plot_utils.plot_output_train_test(model_hdl, train_test_data, 100, False, leg_labels, True, density=True)
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'TopoVarsTrainingCurve_optimizated.pdf')

    plt.clf()
    plot_utils.plot_roc(ytestset, y_pred_test, multi_class_opt="ovo")
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'ROC_testSet_optimizated.pdf')
    
    plt.clf()
    plot_utils.plot_roc(ytrain, y_pred_train, multi_class_opt="ovo")
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'ROC_trainSet_optimizated.pdf')
    
    #Saving the model
    plt.clf()
    plots_shap = plot_utils.plot_feature_imp(train_test_data[2], train_test_data[3], model_hdl, inputCfg['Output']['legLabels'])
    for i in range(0,len(plots_shap)):
        plots_shap[i].savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'featureImp_'+str(i)+'.pdf')
        
    model_hdl.dump_model_handler(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+"ModelHandler_MulticlassDs_pT_2_4.pickle")
    model_hdl.dump_original_model(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+"XGBoostModel_MulticlassDs_pT_2_4.pickle")
    print("starting: Scores - data")
    
    #Apply the model to the data dataSet - Data
    plt.clf()
    yPred_data = model_hdl.predict(hdl_data_app, False)
    df_data_app = hdl_data_app.get_data_frame()
    df_data_app['score'] = yPred_data
    df_data_app['score'].plot(kind='hist', bins=100, alpha=0.8, log=True, figsize=(12, 7), grid=False, density=True, label='score')
    plt.legend()
    plt.show()
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'data_prediction.pdf')
    df_data_app_toSave = df_data_app[['fTagsInvMass', 'fDecayLength', 'fDecayLengthXY', 'fDecayLengthNormalised', 'fTrackDcaXY', 'fCpa', 'fCpaXY', 'score']].copy()
    print("completed score calc.")
    
    #Apply the model to the data dataSet - App
    plt.clf()
    
    print("starting mc scores")
    yPred_mc = model_hdl.predict(hdl_mc_app, False)
    df_mc_app = hdl_mc_app.get_data_frame()
    df_mc_app['score'] = yPred_mc
    df_mc_app['score'].plot(kind='hist', bins=100, alpha=0.8, log=True, figsize=(12, 7), grid=False, density=True, label='score')
    plt.legend()
    plt.draw()
    plt.savefig(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'mc_prediction.pdf')
    df_mc_app_toSave = df_mc_app[['fTagsInvMass', 'fDecayLength', 'fDecayLengthXY', 'fDecayLengthNormalised', 'fTrackDcaXY', 'fCpa', 'fCpaXY', 'fIsSignal', 'score']].copy()
    
    #saving the score-dfs to parquet file .gzip
    print("saving the scores")
    #df_mc_app_toSave.to_parquet(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'kaka_mc_dfScores.gzip', compression='gzip', index=False)
    df_data_app_toSave.to_parquet(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'kaka_data_dfScores.gzip', compression='gzip', index=False)

#ML_DMesons(2,4)
