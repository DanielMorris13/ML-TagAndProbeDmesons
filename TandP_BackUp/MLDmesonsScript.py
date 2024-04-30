import MLDmesonsUtils as MLU
import ROOT as root
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
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
from particle import Particle

from hipe4ml.model_handler import ModelHandler
from hipe4ml.tree_handler import TreeHandler
from hipe4ml import plot_utils
from hipe4ml.analysis_utils import train_test_generator
from hipe4ml_converter.h4ml_converter import H4MLConverter

import uproot
from flarefly.data_handler import DataHandler
from flarefly.fitter import F2MassFitter


########################################################################################################################################
#Produce i le tabelle con le var- topologiche + Score per un fissato PT-bin -> Loop su tutti i PT-bin
def Gen_score_PT_dif():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgFileName', default='Config_MLDmesonsScript.yml', help='config file name for ml')
    args = parser.parse_args()

    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    ptmin_list = [0,1,2,4,6,10,20]
    ptmax_list = [1,2,4,6,10,20,100]

    for i in range(2,3):
        print("runnign on PT range", ptmin_list[i], ptmax_list[i])
        MLU.ML_DMesons(ptmin_list[i],ptmax_list[i])

########################################################################################################################################
#produce la distribuzione degli score 
def score_plot():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgFileName', default='Config_MLDmesonsScript.yml', help='config file name for ml')
    args = parser.parse_args()

    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
        
    hdl_data = TreeHandler(inputCfg['Output']['dir']+'mlData/'+'kaka_mc_dfScores.gzip')
    df_data = hdl_data.get_data_frame()   
    
    ofile = root.TFile(inputCfg['pyroot']['plotsDir']+'plots_score_mc.root', 'recreate')
    cScore = root.TCanvas()
    
    hscore = root.TH1F("hscore", ";score;# Entries", 500, 0, 1)
    
    for x in df_data['score']:
        hscore.Fill(x)
    
    hscore.Draw()
    cScore.SaveAs(inputCfg['pyroot']['plotsDir']+'score.pdf')
    ofile.Write()
    
    
########################################################################################################################################   
#genera i file .root con tutte le distribuzione di massa
def CutOpt_method2(Pcut, PTmin, PTmax):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgFileName', default='Config_MLDmesonsScript.yml', help='config file name for ml')
    args = parser.parse_args()

    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    
    hdl_data = TreeHandler(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+'kaka_data_dfScores.gzip')
    df_data = hdl_data.get_data_frame()
     
    ofile = root.TFile(inputCfg['Output']['dirPT']+f'PT_{PTmin}_{PTmax}_test/'+f'plots_InvMass_pt_{PTmin}_{PTmax}.root', 'update')

    canvas = root.TCanvas()
    hselected = root.TH1F(f'hselected_{Pcut}', 'Tags invariant mass; M(K^{+}K^{-}) (GeV)', 250,0.98, 1.06)
    #h = root.TH1F(f'h_{Pcut}', "Tags invariant mass; M(K^{+}K^{-}) (GeV)", 250,0.98, 1.06)
    
    
    print("before cut",len(df_data))

    df_data.query(f'score > {Pcut}', inplace=True)
    #df_data.query(f'{PTmin} < fTagsPt < {PTmax}',  inplace=True)

    print("after selection",len(df_data))
    
    for mass in df_data['fTagsInvMass']:
        hselected.Fill(mass)
    
    '''
    for mass in df_data['fTagsInvMass']:
        h.Fill(mass)
    '''
    canvas.cd()
    hselected.Draw()
    #canvas.SaveAs(inputCfg['pyroot']['plotsDir']+f'InvMassDist_Cut_{Pcut}.pdf')
    ofile.Write()
    ofile.Close()
    
########################################################################################################################################   
#Genera la distribuzione di massa per fissato PTbin e score 
def Fitting(Pcut, PTmin, PTmax):
    data = DataHandler(data='/home/dmorris/ML-TagAndProbeDmesons/outputs/PT_'+f'{PTmin}_{PTmax}_test/'+f'plots_InvMass_pt_{PTmin}_{PTmax}.root', 
                       var_name=r"$M_\mathrm{D^{-}\pi^{+}}$ (GeV/$c^{2}$)", histoname=f'hselected_{Pcut}',
                       limits=[0.995, 1.05])
    
    # define PDFs lists
    signal_pdfs = ["voigtian"]
    background_pdfs = ["chebpol2"]

    # define the ids
    voigtian_id = 0 # because signal_pdfs[gaussian_id] = "gaussian"
    expo_id = 0     # because background_pdfs[expo_id] = "expo"

    fitter = F2MassFitter(data_handler=data,
                      name_signal_pdf=signal_pdfs,
                      name_background_pdf=background_pdfs,
                      name=f"{background_pdfs[expo_id]}_{signal_pdfs[voigtian_id]}_{Pcut}")
    # set the initial parameters 
    
    ''' 2_4
    mass = Particle.from_pdgid(333).mass*1e-3
    width = Particle.from_pdgid(333).width*1.e-3
    #print(mass, width)
    fitter.set_particle_mass(0, mass=mass, limits=[mass-0.0005, mass+0.005])
    fitter.set_signal_initpar(0, "gamma", width, limits=[0,0.01])
        
    fitter.set_signal_initpar(voigtian_id, "frac", 0.2, limits=[0.0, 1.0])
    fitter.set_signal_initpar(voigtian_id, "sigma", 0.005, limits=[0, 0.01])#Gev
    fitter.set_background_initpar(0, "c0", 5)
    fitter.set_background_initpar(0, "c1", 1)
    fitter.set_background_initpar(0, "c2", -0.1)
    '''
    mass = Particle.from_pdgid(333).mass*1e-3
    width = Particle.from_pdgid(333).width*1.e-3
    #print(mass, width)
    fitter.set_particle_mass(0, mass=mass, limits=[mass-0.0005, mass+0.005])
    fitter.set_signal_initpar(0, "gamma", width, limits=[0,0.01])
        
    fitter.set_signal_initpar(voigtian_id, "frac", 0.3231, limits=[0.0, 1.0])
    fitter.set_signal_initpar(voigtian_id, "sigma", 0.000130734, limits=[0, 0.03])#Gev
    fitter.set_signal_initpar(voigtian_id, "mu", mass, limits=[mass-0.005, mass+0.005])
    fitter.set_background_initpar(0, "c0", 5.17797)
    fitter.set_background_initpar(0, "c1",  1.03876)
    fitter.set_background_initpar(0, "c2", -0.198126 )
    
    ''' 6_10
    fitter.set_particle_mass(0, mass=mass, limits=[mass-0.005, mass+0.005])
    fitter.set_signal_initpar(0, "gamma", width, limits=[0,0.01])
        
    fitter.set_signal_initpar(voigtian_id, "frac", 0.3, limits=[0.0, 1.0])
    fitter.set_signal_initpar(voigtian_id, "sigma", 0.0005, limits=[0, 0.01])#Gev
    fitter.set_background_initpar(0, "c0", 5)
    fitter.set_background_initpar(0, "c1", 1)
    fitter.set_background_initpar(0, "c2", -0.1)
    '''
    #fitter.set_background_initpar(expo_id, "lam", -0.3)
    fitter.mass_zfit()
    # plot the fit result with display options
    plot_mass_fit = fitter.plot_mass_fit(style="ATLAS",
                    show_extra_info=True,
                    extra_info_loc=['upper left', 'center right'])
    
    #fitter.dump_to_root(filename="/home/dmorris/ML-TagAndProbeDmesons/outputs/PyRootPlots/outputFit.root")
    fig_reso = fitter.plot_mass_fit(style="ATLAS",
                                             figsize=(8, 8),
                                             axis_title=rf"M (GeV/$c^2$)",
                                             show_extra_info=True)
    fig_reso_res = fitter.plot_raw_residuals(style="ATLAS", figsize=(8, 8), axis_title=rf"M (GeV/$c^2$)")
    
    p = PdfPages(f'/home/dmorris/ML-TagAndProbeDmesons/outputs/PT_'+f'{PTmin}_{PTmax}_test/'+f'Fit_{Pcut}.pdf')
    
    fig_reso.savefig(p,format='pdf')
    fig_reso_res.savefig(p,format='pdf')
    
    p.close()
    Sb = fitter.get_signal_over_background(0, nhwhm=3.)
    print("S/B: ", Sb)

    #fig_reso.savefig(f"mass_{Pcut}.pdf")
    #fig_reso_res.savefig(f"mass_res_{Pcut}.pdf")
    #fitter.dump_to_root(root.Form("outputFit.root",s), option="update")

########################################################################################################################################
#Genera le distribuzioni di masse invariante per tutti i cut di score in un PT bin
def LoopOverPcut(PTmin,PTmax):
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgFileName', default='Config_MLDmesonsScript.yml', help='config file name for ml')
    args = parser.parse_args()

    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    
    Pcuts = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.96,0.97,0.98,0.99]
    for Pcut in Pcuts:
        print(Pcut)
        CutOpt_method2(Pcut,PTmin,PTmax)
        
    
########################################################################################################################################
def LoopOverFits(PTmin,PTmax): 
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfgFileName', default='Config_MLDmesonsScript.yml', help='config file name for ml')
    args = parser.parse_args()
    
    with open(args.cfgFileName, 'r') as ymlCfgFile:
        inputCfg = yaml.load(ymlCfgFile, yaml.FullLoader)
    
    Pcuts = [0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.96,0.97]

    for Pcut in Pcuts:
        ifile = root.TFile.Open(f'/home/dmorris/ML-TagAndProbeDmesons/outputs/PT_{PTmin}_{PTmax}/plots_InvMass_pt_{PTmin}_{PTmax}.root',"read")
        print("In process: ",Pcut)
        Fitting(Pcut,PTmin,PTmax)
    
########################################################################################################################################
def SB_Pcut(PTmin,PTmax):
    Pcuts =                           [0.50,      0.55,       0.60,       0.65,      0.70,        0.75,       0.80,       0.85,       0.90,        0.95,       0.96,       0.97]
    SignalBackground_6_10 =           [0.7407,    0.7307,     0.7276,     0.7309,    0.6739,      0.7249,     0.7314,     0.7564,     0.7367,      0.8487,     0.5266,     0.6821]
    err_SignalBackground_6_10 =       [0.0285,    0.0325,     0.0638,     0.0686,    0.0909,      0.0309,     0.1164,     0.0517,     0.0925,      0.1877,     0.2731,     0.2626] 
    SignalBackground_2_4 =            [0.2663,    0.2674,     0.2720,     0.3057,    0.3104,      0.3176,     0.3269,     0.3317,     0.2906,      0.3814,     0.32897,     0.3238]
    err_SignalBackground_2_4 =        [0.0081,    0.0086,     0.0097,     0.0059,    0.00652,     0.0074,     0.0093,     0.0108,     0.0250,      0.0237,    0.04678,      0.0466]
    
    Pcuts_test =                      [0.50,      0.55,       0.60,       0.65,      0.70,        0.75,       0.80, 0.85]
    SignalBackground_2_4_test =       [0.2671,    0.2877,     0.3123,     0.2944,    0.3244,      0.29797,    0.3010   ]
    err_SignalBackground_2_4_test =   [0.018,     0.0269,     0.0183,     0.0333,    0.0331,      0.04884,    0.1547   ]
    
    SignalBackground_6_10_test =       [0.7229,    0.7265,   0.7463,     0.7707,     0.7709,       0.7523,     0.7439,     0.7636     ]
    err_SignalBackground_6_10_test =   [0.0330,    0.0262,   0.0318,     0.0641,     0.0456,       0.05295,    0.0682,     0.1016   ]    

    plt.errorbar(Pcuts, SignalBackground_2_4, yerr=err_SignalBackground_2_4, fmt="o")
    plt.xlabel("Score cut (a.u)")
    plt.ylabel("S/B (3HWHM)")

    plt.show()
    plt.savefig(f'/home/dmorris/ML-TagAndProbeDmesons/outputs/PT_'+f'{PTmin}_{PTmax}/'+'SB_Pcut.pdf')
    print(f'/home/dmorris/ML-TagAndProbeDmesons/outputs/PT_'+f'{PTmin}_{PTmax}_test/'+'SB_Pcut.pdf')
 ########################################################################################################################################
   
#binsPt = {0., 1., 2., 4., 6., 10., 20., 1000.}

#LoopOverPcut(6,10)
#Gen_score_PT_dif()z
#Fitting(0.9, 6, 10)
#LoopOverFits(6, 10)
#print("Check")
SB_Pcut(2,4)


