import ROOT as root
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


def plots_maker():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgFileName', default='Config_MLDmesonsScript.yml', help='config file name for ml')
    args = parser.parse_args()

    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    
    hdl_mc_ml = TreeHandler(inputCfg['Output']['dir']+'kaka_mc_dataset_ml_80.gzip')
    hdl_data_app = TreeHandler(inputCfg['Output']['dir']+'kaka_data_dataset_app.gzip')

    df_mc_ml = hdl_mc_ml.get_data_frame()
    df_data_app = hdl_data_app.get_data_frame()
    
    cDecayLength = root.TCanvas()
    hDecayLength = root.TH1F("hDecayLength", "DecayLength; DecayLength (cm); # Entries", 250, 0, 0.5)
    
    for x in df_mc_ml['fDecayLength']:
        hDecayLength.Fill(x)
    
    hDecayLength.Draw()
    cDecayLength.SaveAs(inputCfg['pyroot']['plotsDir']+'DecayLength.pdf')

plots_maker()  
    
    
    