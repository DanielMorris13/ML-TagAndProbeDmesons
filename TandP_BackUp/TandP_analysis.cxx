void TandP_analysis(TString inFileNameMc,TString inFileNameData){
  
  const int pTmin = 0;
  const int pTmax = 100;

  TFile* iFileMc = new TFile(inFileNameMc+".root","READ");
  TFile* iFileData = new TFile(inFileNameData+".root","READ");

  TTree* iTreeData = (TTree*)iFileData->Get("O2topovars");
  TTree* iTreeMc = (TTree*)iFileMc->Get("O2topovars");

  TFile* ofile = new TFile(Form(inFileNameMc+"_%i_to_%i_pt_plots.root",pTmin,pTmax),"RECREATE");

  typedef struct {
    float Pt;
    float InvMass;
    float DecayLength;
    float DecayLengthXY;
    float DecayLengthNormalized;
    float TrackDcaXY;
    float Cpa;
    float CpaXY;
    int IsSignal;
  } TagInfo;

  static TagInfo tagMc;
  static TagInfo tagData;
  TCanvas* cInvMass = new TCanvas("cInvMass","InvMass",2560,1600);
  TCanvas* cDecayLength = new TCanvas("cDecayLength","DecayLength",2560,1600);
  TCanvas* cDecayLengthXY = new TCanvas("cDecayLengthXY","DecayLengthXY",2560,1600); 
  TCanvas* cDecayLengthNormalised= new TCanvas("cDecayLengthNormalised","DecayLengthNormalised",2560,1600);
  TCanvas* cTrackDcaXY = new TCanvas("cTrackDcaXY","TrackDcaXY",2560,1600);
  TCanvas* cCpa = new TCanvas("cCpa","Cpa",2560,1600);
  TCanvas* cCpaXY = new TCanvas("cCpaXY","CpaXY",2560,1600);

  //Binnign per KK
  const int InvMass_bins = 150;
  const float InvMass_xlow = 0.987;  //geV
  const float InvMass_xup = 1.060;   //gev

  const int DecayLength_bins = 250;
  const int DecayLength_xlow = 0;
  const double DecayLength_xup = 0.6;

  const int DecayLengthXY_bins = 125;
  const int DecayLengthXY_xlow = 0;
  const double DecayLengthXY_xup = 0.25;

  const int DecayLengthNormalized_bins = 200;
  const int DecayLengthNormalized_xlow = 0;
  const int DecayLengthNormalized_xup = 400;

  const int TrackDcaXY_bins = 250;
  const int TrackDcaXY_xlow = 0;
  const double TrackDcaXY_xup = 0.00025;
  
  const int Cpa_bins = 150;
  const double Cpa_low = 0.98;
  const int Cpa_up = 1;

/*
 //Binning KPi
  const int InvMass_bins = 150;
  const float InvMass_xlow = 1.65;  //geV
  const float InvMass_xup =  2.05;   //gev

  const int DecayLength_bins = 250;
  const int DecayLength_xlow = 0;
  const double DecayLength_xup = 0.2;

  const int DecayLengthXY_bins = 250;
  const int DecayLengthXY_xlow = 0;
  const double DecayLengthXY_xup = 0.2;

  const int DecayLengthNormalized_bins = 200;
  const int DecayLengthNormalized_xlow = 0;
  const int DecayLengthNormalized_xup = 20;

  const int TrackDcaXY_bins = 250;
  const int TrackDcaXY_xlow = 0;
  const double TrackDcaXY_xup = 0.0001;
  
  const int Cpa_bins = 150;
  const double Cpa_low = 0.98;
  const int Cpa_up = 1;
*/

 /*//Binnig PiPi
  const int InvMass_bins = 150;
  const float InvMass_xlow = 0.28;  //geV
  const float InvMass_xup =  1.52;   //gev

  const int DecayLength_bins = 250;
  const int DecayLength_xlow = 0;
  const double DecayLength_xup = 0.6;

  const int DecayLengthXY_bins = 250;
  const int DecayLengthXY_xlow = 0;
  const double DecayLengthXY_xup = 0.25;

  const int DecayLengthNormalized_bins = 400;
  const int DecayLengthNormalized_xlow = 0;
  const int DecayLengthNormalized_xup = 150;

  const int TrackDcaXY_bins = 250;
  const int TrackDcaXY_xlow = 0;
  const double TrackDcaXY_xup = 0.00025;
  
  const int Cpa_bins = 150;
  const double Cpa_low = 0.98;
  const int Cpa_up = 1;
*/

  /*-------------------------------------------------------MC------------------------------------------------------------------*/
  THStack *sInvMass = new THStack("sInvMass","Invariant Mass;M(K^{#pm} K^{#mp})[GeV];# Normalized Counts");
  THStack *sDecayLength = new THStack("sDecayLength","Decay Length; decLen[cm];# Normalized Counts");
  THStack* sDecayLengthXY = new THStack("hDecayLengthXY","Decay Length XY;decLen_{XY}[cm];# Normalized Counts");
  THStack* sDecayLengthNormalised = new THStack("sDecayLengthNormalised","Decay Length Normalised;decLen_{N};# Normalized Counts");
  THStack* sTrackDcaXY = new THStack("sTrackDcaXYMc","TrackDca_{XY};[cm^{2}];# Normalized Counts");
  THStack* sCpa = new THStack("sCpaMc","CPA;cos(#theta_{PA});# Normalized Counts");
  THStack* sCpaXY = new THStack("sCpaXY","Cpa_{XY};cos(#theta_{PA});# Normalized Counts");

  TH1F* hInvMassMc = new TH1F("hInvMassMc","hInvMassMc; M(K^{#pm}K^{#mp})[GeV^{2}];# Entries ", InvMass_bins,InvMass_xlow,InvMass_xup);
  TH1F* hDecayLengthMc = new TH1F("hDecayLengthMc","hDecayLength;[cm];#Entries",DecayLength_bins,DecayLength_xlow,DecayLength_xup);
  TH1F* hDecayLengthXYMc = new TH1F("hDecayLengthXYMc","hDecayLengthXY;[cm];#Entries",DecayLengthXY_bins,DecayLengthXY_xlow,DecayLengthXY_xup);
  TH1F* hDecayLengthNormalisedMc = new TH1F("hDecayLengthNormalisedMc","hDecayLengthNormalised;[cm];#Entries",DecayLengthNormalized_bins,DecayLengthNormalized_xlow,DecayLengthNormalized_xup);
  TH1F* hTrackDcaXYMc = new TH1F("hTrackDcaXYMc","hTrackDcaXY;[cm];#Entries",TrackDcaXY_bins,TrackDcaXY_xlow,TrackDcaXY_xup);
  TH1F* hCpaMc = new TH1F("hCpaMc","hCpa;cos(#theta_{PA});#Entries",Cpa_bins,Cpa_low,Cpa_up);
  TH1F* hCpaXYMc = new TH1F("hCpaXYMc","hCpaXY;cos(#theta_{PA}^{XY});#Entries",Cpa_bins,Cpa_low,Cpa_up);

  TH1F* hInvMassSignal = new TH1F("hInvMassSignal","hInvMassSignal; M(K^{#pm}K^{#mp})[GeV^{2}];# Entries ", InvMass_bins,InvMass_xlow,InvMass_xup);
  TH1F* hDecayLengthSignal = new TH1F("hDecayLengthSignal","hDecayLengthSignal",DecayLength_bins,DecayLength_xlow,DecayLength_xup);
  TH1F* hDecayLengthXYSignal= new TH1F("hDecayLengthXYSignal","hDecayLengthSignal",DecayLengthXY_bins,DecayLengthXY_xlow,DecayLengthXY_xup);
  TH1F* hDecayLengthNormalisedSignal= new TH1F("hDecayLengthNormalisedSignal","hDecayLengthNormalisedSignal",DecayLengthNormalized_bins,DecayLengthNormalized_xlow,DecayLengthNormalized_xup);
  TH1F* hTrackDcaXYSignal= new TH1F("hTrackDcaXYSignal","hTrackDcaXYSignal",TrackDcaXY_bins,TrackDcaXY_xlow,TrackDcaXY_xup);
  TH1F* hCpaSignal= new TH1F("hCpaSignal","hCpaSignal",Cpa_bins,Cpa_low,Cpa_up);
  TH1F* hCpaXYSignal= new TH1F("hCpaXYSignal","hCpaXYSignal",Cpa_bins,Cpa_low,Cpa_up);

  hInvMassSignal->SetLineColor(2);
  hDecayLengthSignal->SetLineColor(2);
  hDecayLengthXYSignal->SetLineColor(2);
  hDecayLengthNormalisedSignal->SetLineColor(2);
  hTrackDcaXYSignal->SetLineColor(2);
  hCpaSignal->SetLineColor(2);
  hCpaXYSignal->SetLineColor(2);
/*
  TH1F* hInvMassSignalNPrompt = new TH1F("hInvMassSignalNPrompt","hInvMassSignalNPrompt; M(K^{#pm}K^{#mp})[GeV^{2}];# Entries ", InvMass_bins,InvMass_xlow,InvMass_xup);
  TH1F* hDecayLengthSignalNPrompt = new TH1F("hDecayLengthSignalNPrompt","hDecayLengthSignalNPrompt",DecayLength_bins,DecayLength_xlow,DecayLength_xup);
  TH1F* hDecayLengthXYSignalNPrompt = new TH1F("hDecayLengthXYSignalNPrompt","hDecayLengthSignalNPrompt",DecayLengthXY_bins,DecayLengthXY_xlow,DecayLengthXY_xup);
  TH1F* hDecayLengthNormalisedSignalNPrompt = new TH1F("hDecayLengthNormalisedSignalNPrompt","hDecayLengthNormalisedSignalNPrompt",DecayLengthNormalized_bins,DecayLengthNormalized_xlow,DecayLengthNormalized_xup);
  TH1F* hTrackDcaXYSignalNPromt = new TH1F("hTrackDcaXYSignalNPromt","hTrackDcaXYSignalNPromt",TrackDcaXY_bins,TrackDcaXY_xlow,TrackDcaXY_xup);
  TH1F* hCpaSignalNPrompt = new TH1F("hCpaSignalNPrompt","hCpaSignalNPrompt",Cpa_bins,Cpa_low,Cpa_up);
  TH1F* hCpaXYSignalNPrompt = new TH1F("hCpaXYSignalNPrompt","hCpaXYSignalNPrompt",Cpa_bins,Cpa_low,Cpa_up);

  hInvMassSignalNPrompt->SetLineColor(3);
  hDecayLengthSignalNPrompt->SetLineColor(3);
  hDecayLengthXYSignalNPrompt->SetLineColor(3);
  hDecayLengthNormalisedSignalNPrompt->SetLineColor(3);
  hTrackDcaXYSignalNPromt->SetLineColor(3);
  hCpaSignalNPrompt->SetLineColor(3);
  hCpaXYSignalNPrompt->SetLineColor(3);

  TH1F* hInvMassSignalPrompt = new TH1F("hInvMassSignalPrompt","hInvMassSignalPrompt; M(K^{#pm}K^{#mp})[GeV^{2}];# Entries ", InvMass_bins,InvMass_xlow,InvMass_xup);
  TH1F* hDecayLengthSignalPrompt = new TH1F("hDecayLengthSignalPrompt","hDecayLengthSignalPrompt",DecayLength_bins,DecayLength_xlow,DecayLength_xup);
  TH1F* hDecayLengthXYSignalPrompt = new TH1F("hDecayLengthXYSignalPrompt","hDecayLengthSignalPrompt",DecayLengthXY_bins,DecayLengthXY_xlow,DecayLengthXY_xup);
  TH1F* hDecayLengthNormalisedSignalPrompt = new TH1F("hDecayLengthNormalisedSignalPrompt","hDecayLengthNormalisedSignalPrompt",DecayLengthNormalized_bins,DecayLengthNormalized_xlow,DecayLengthNormalized_xup);
  TH1F* hTrackDcaXYSignalPromt = new TH1F("hTrackDcaXYSignalPromt","hTrackDcaXYSignalNPromt",TrackDcaXY_bins,TrackDcaXY_xlow,TrackDcaXY_xup);
  TH1F* hCpaSignalPrompt = new TH1F("hCpaSignalPrompt","hCpaSignalPrompt",Cpa_bins,Cpa_low,Cpa_up);
  TH1F* hCpaXYSignalPrompt = new TH1F("hCpaXYSignalPrompt","hCpaXYSignalPrompt",Cpa_bins,Cpa_low,Cpa_up);

  hInvMassSignalPrompt->SetLineColor(1);
  hDecayLengthSignalPrompt->SetLineColor(1);
  hDecayLengthXYSignalPrompt->SetLineColor(1);
  hDecayLengthNormalisedSignalPrompt->SetLineColor(1);
  hTrackDcaXYSignalPromt->SetLineColor(1);
  hCpaSignalPrompt->SetLineColor(1);
  hCpaXYSignalPrompt->SetLineColor(1);
*/
  iTreeMc->SetBranchAddress("fTagsPt",&tagMc.Pt);
  iTreeMc->SetBranchAddress("fTagsInvMass",&tagMc.InvMass);
  iTreeMc->SetBranchAddress("fDecayLength",&tagMc.DecayLength);
  iTreeMc->SetBranchAddress("fDecayLengthXY",&tagMc.DecayLengthXY);
  iTreeMc->SetBranchAddress("fDecayLengthNormalised",&tagMc.DecayLengthNormalized);
  iTreeMc->SetBranchAddress("fTrackDcaXY",&tagMc.TrackDcaXY);
  iTreeMc->SetBranchAddress("fCpa",&tagMc.Cpa);
  iTreeMc->SetBranchAddress("fCpaXY",&tagMc.CpaXY);
  iTreeMc->SetBranchAddress("fIsSignal",&tagMc.IsSignal);
  //iTreeMc->SetBranchAddress("fIsPrompt",&tagMc.IsPrompt);


  /*-------------------------------------------------------Data----------------------------------------------------------------*/
  TH1F* hInvMassData = new TH1F("hInvMassData","hInvMassData; M(K^{#pm}K^{#mp})[GeV}];# Entries ", InvMass_bins,InvMass_xlow,InvMass_xup);
  TH1F* hDecayLengthData = new TH1F("hDecayLengthData","hDecayLength;[cm];#Entries",DecayLength_bins,DecayLength_xlow,DecayLength_xup);
  TH1F* hDecayLengthXYData = new TH1F("hDecayLengthXYData","hDecayLengthXY;[cm];#Entries",DecayLengthXY_bins,DecayLengthXY_xlow,DecayLengthXY_xup);
  TH1F* hDecayLengthNormalisedData = new TH1F("hDecayLengthNormalisedData","hDecayLengthNormalised;[cm];#Entries",DecayLengthNormalized_bins,DecayLengthNormalized_xlow,DecayLengthNormalized_xup);
  TH1F* hTrackDcaXYData = new TH1F("hTrackDcaXYData","hTrackDcaXY;[cm];#Entries",TrackDcaXY_bins,TrackDcaXY_xlow,TrackDcaXY_xup);
  TH1F* hCpaData = new TH1F("hCpaData","hCpa;cos(#theta_{PA});#Entries",Cpa_bins,Cpa_low,Cpa_up);
  TH1F* hCpaXYData = new TH1F("hCpaXYData","hCpa;cos(#theta_{PA});#Entries",Cpa_bins,Cpa_low,Cpa_up);

  hInvMassData->SetLineColor(6);
  hDecayLengthData->SetLineColor(6);
  hDecayLengthXYData->SetLineColor(6);
  hDecayLengthNormalisedData->SetLineColor(6);
  hTrackDcaXYData->SetLineColor(6);
  hCpaData->SetLineColor(6);
  hCpaXYData->SetLineColor(6);

  iTreeData->SetBranchAddress("fTagsPt",&tagData.Pt);
  iTreeData->SetBranchAddress("fTagsInvMass",&tagData.InvMass);
  iTreeData->SetBranchAddress("fDecayLength",&tagData.DecayLength);
  iTreeData->SetBranchAddress("fDecayLengthXY",&tagData.DecayLengthXY);
  iTreeData->SetBranchAddress("fDecayLengthNormalised",&tagData.DecayLengthNormalized);
  iTreeData->SetBranchAddress("fTrackDcaXY",&tagData.TrackDcaXY);
  iTreeData->SetBranchAddress("fCpa",&tagData.Cpa);
  iTreeData->SetBranchAddress("fCpaXY",&tagData.CpaXY);
  iTreeData->SetBranchAddress("fIsSignal",&tagData.IsSignal);
  //iTreeData->SetBranchAddress("fIsPrompt",&tagData.IsPrompt);
  
  /*---------------------------------------------------------------------------------------------------------------------------*/ 
  sInvMass->Add(hInvMassData);
  sInvMass->Add(hInvMassMc);
  sInvMass->Add(hInvMassSignal);
  //sInvMass->Add(hInvMassSignalNPrompt);
  //sInvMass->Add(hInvMassSignalPrompt);

  sDecayLength->Add(hDecayLengthData);
  sDecayLength->Add(hDecayLengthMc);
  sDecayLength->Add(hDecayLengthSignal);
  //sDecayLength->Add(hDecayLengthSignalNPrompt);
  //sDecayLength->Add(hDecayLengthSignalPrompt);

  sDecayLengthNormalised->Add(hDecayLengthNormalisedData);
  sDecayLengthNormalised->Add(hDecayLengthNormalisedMc);
  sDecayLengthNormalised->Add(hDecayLengthNormalisedSignal);
  //sDecayLengthNormalised->Add(hDecayLengthNormalisedSignalNPrompt);
  //sDecayLengthNormalised->Add(hDecayLengthNormalisedSignalPrompt);

  sDecayLengthXY->Add(hDecayLengthXYData);
  sDecayLengthXY->Add(hDecayLengthXYMc);
  sDecayLengthXY->Add(hDecayLengthXYSignal);
  //sDecayLengthXY->Add(hDecayLengthXYSignalNPrompt);
  //sDecayLengthXY->Add(hDecayLengthXYSignalPrompt);

  sTrackDcaXY->Add(hTrackDcaXYData);
  sTrackDcaXY->Add(hTrackDcaXYMc);
  sTrackDcaXY->Add(hTrackDcaXYSignal);
  //sTrackDcaXY->Add(hTrackDcaXYSignalNPromt);
  //sTrackDcaXY->Add(hTrackDcaXYSignalPromt);

  sCpa->Add(hCpaData);
  sCpa->Add(hCpaMc);
  sCpa->Add(hCpaSignal);
  //sCpa->Add(hCpaSignalNPrompt);
  //sCpa->Add(hCpaSignalPrompt);

  sCpaXY->Add(hCpaXYData);
  sCpaXY->Add(hCpaXYMc);
  sCpaXY->Add(hCpaXYSignal);
  //sCpaXY->Add(hCpaXYSignalNPrompt);
  //sCpaXY->Add(hCpaXYSignalPrompt);

  
 /*---------------------------------------------------------------------------------------------------------------------------*/

  for( auto ev = 0 ; ev < iTreeMc->GetEntries() ; ev++ ){
    iTreeMc->GetEvent(ev);
    
    if( tagMc.IsSignal != 1 && tagMc.IsSignal != -999 ){ //blu
      if( tagMc.Pt < pTmin || tagMc.Pt > pTmax ) continue;

      hInvMassMc->Fill(tagMc.InvMass);
      hDecayLengthMc->Fill(tagMc.DecayLength);
      hDecayLengthXYMc->Fill(tagMc.DecayLengthXY);
      hDecayLengthNormalisedMc->Fill(tagMc.DecayLengthNormalized);
      hTrackDcaXYMc->Fill(tagMc.TrackDcaXY);
      hCpaMc->Fill(tagMc.Cpa);
      hCpaXYMc->Fill(tagMc.CpaXY);
    }

    if( tagMc.IsSignal == 1  ){ //rosso

      if( tagMc.Pt < pTmin || tagMc.Pt > pTmax ) continue;

      hInvMassSignal->Fill(tagMc.InvMass);
      hDecayLengthSignal->Fill(tagMc.DecayLength);
      hDecayLengthXYSignal->Fill(tagMc.DecayLengthXY);
      hDecayLengthNormalisedSignal->Fill(tagMc.DecayLengthNormalized);
      hTrackDcaXYSignal->Fill(tagMc.TrackDcaXY);
      hCpaSignal->Fill(tagMc.Cpa);
      hCpaXYSignal->Fill(tagMc.CpaXY);
      
    }
  }

  for( auto ev = 0 ; ev < iTreeData->GetEntries() ; ev++ ){
  
    iTreeData->GetEvent(ev);
    
    if( tagData.Pt < pTmin || tagData.Pt > pTmax ) continue;

    hInvMassData->Fill(tagData.InvMass);
    hDecayLengthData->Fill(tagData.DecayLength);
    hDecayLengthXYData->Fill(tagData.DecayLengthXY);
    hDecayLengthNormalisedData->Fill(tagData.DecayLengthNormalized);
    hTrackDcaXYData->Fill(tagData.TrackDcaXY);
    hCpaData->Fill(tagData.Cpa);
    hCpaXYData->Fill(tagData.CpaXY);
  }


  cDecayLength->cd();
  auto legend = new TLegend(0.6,0.7,0.89,0.89);
  legend->AddEntry(hDecayLengthData,"Data","F");
  legend->AddEntry(hDecayLengthMc,"X #rightarrow (K^{#pm} K^{#mp})_{Tag}","F");
  legend->AddEntry(hDecayLengthSignal,"D_{s}^{#pm} / D^{#pm} #rightarrow #Phi #pi^{#pm} #rightarrow (K^{#pm} K^{#mp})_{Tag} #pi^{#pm}","F");
  //legend->AddEntry(hDecayLengthSignalNPrompt,"D^{*#pm}/D_{s}^{#pm} #rightarrow #Phi #rightarrow K^{#pm} K^{#mp}","F");
  //legend->AddEntry(hDecayLengthSignalPrompt,"X #rightarrow #Phi #rightarrow K^{#pm} K^{#mp}","F");

  hDecayLengthMc->Scale(1./hDecayLengthMc->GetEntries());
  hDecayLengthData->Scale(1./hDecayLengthData->GetEntries());
  hDecayLengthSignal->Scale(1./hDecayLengthSignal->GetEntries());
  //hDecayLengthSignalNPrompt->Scale(1./hDecayLengthSignalNPrompt->GetEntries());
  //hDecayLengthSignalPrompt->Scale(1./hDecayLengthSignalPrompt->GetEntries());

  sDecayLength->Draw("nostackhisto");
  legend->Draw();

  cDecayLengthXY->cd();
  hDecayLengthXYMc->Scale(1./hDecayLengthXYMc->GetEntries());
  hDecayLengthXYData->Scale(1./hDecayLengthXYData->GetEntries());
  hDecayLengthXYSignal->Scale(1./hDecayLengthXYSignal->GetEntries());
  //hDecayLengthXYSignalNPrompt->Scale(1./hDecayLengthXYSignalNPrompt->GetEntries());
  //hDecayLengthXYSignalPrompt->Scale(1./hDecayLengthXYSignalPrompt->GetEntries());
  sDecayLengthXY->Draw("nostackhisto");
  legend->Draw();


  cDecayLengthNormalised->cd();
  hDecayLengthNormalisedMc->Scale(1./hDecayLengthNormalisedMc->GetEntries());
  hDecayLengthNormalisedData->Scale(1./hDecayLengthNormalisedData->GetEntries());
  hDecayLengthNormalisedSignal->Scale(1./hDecayLengthNormalisedSignal->GetEntries());
  //hDecayLengthNormalisedSignalNPrompt->Scale(1./hDecayLengthNormalisedSignalNPrompt->GetEntries());
  //hDecayLengthNormalisedSignalPrompt->Scale(1./hDecayLengthNormalisedSignalPrompt->GetEntries());
  sDecayLengthNormalised->Draw("nostackhisto");
  legend->Draw();


  cTrackDcaXY->cd();
  hTrackDcaXYMc->Scale(1./hTrackDcaXYMc->GetEntries());
  hTrackDcaXYData->Scale(1./hTrackDcaXYData->GetEntries());
  hTrackDcaXYSignal->Scale(1./hTrackDcaXYSignal->GetEntries());
  //hTrackDcaXYSignalNPromt->Scale(1./hTrackDcaXYSignalNPromt->GetEntries());
  //hTrackDcaXYSignalPromt->Scale(1./hTrackDcaXYSignalPromt->GetEntries());
  sTrackDcaXY->Draw("nostackhisto");
  legend->Draw();


  cCpa->cd();
  hCpaMc->Scale(1./hCpaMc->GetEntries());
  hCpaData->Scale(1./hCpaData->GetEntries());
  hCpaSignal->Scale(1./hCpaSignal->GetEntries());
  //hCpaSignalNPrompt->Scale(1./hCpaSignalNPrompt->GetEntries());
  //hCpaSignalPrompt->Scale(1./hCpaSignalPrompt->GetEntries());
  sCpa->Draw("nostackhisto");  
  legend->Draw();


  cCpaXY->cd();
  hCpaXYMc->Scale(1./hCpaXYMc->GetEntries());
  hCpaXYData->Scale(1./hCpaXYData->GetEntries());
  hCpaXYSignal->Scale(1./hCpaXYSignal->GetEntries());
  //hCpaXYSignalNPrompt->Scale(1./hCpaXYSignalNPrompt->GetEntries());
  //hCpaXYSignalPrompt->Scale(1./hCpaXYSignalPrompt->GetEntries());
  sCpaXY->Draw("nostackhisto");
  legend->Draw();

  cInvMass->cd();
  hInvMassMc->Scale(1./hInvMassMc->GetEntries());
  hInvMassData->Scale(1./hInvMassData->GetEntries());
  hInvMassSignal->Scale(1./hInvMassSignal->GetEntries());
  //hInvMassSignalNPrompt->Scale(1./hInvMassSignalNPrompt->GetEntries());
  //hInvMassSignalPrompt->Scale(1./hInvMassSignalPrompt->GetEntries());

  sInvMass->Draw("nostackhisto");
  legend->Draw();
  
  cInvMass->Print(Form("/home/dmorris/KaKa_%i_to_%i_pt.pdf(",pTmin,pTmax),"pdf");
  cDecayLength->Print(Form("/home/dmorris/KaKa_%i_to_%i_pt.pdf",pTmin,pTmax),"pdf");
  cDecayLengthXY->Print(Form("/home/dmorris/KaKa_%i_to_%i_pt.pdf",pTmin,pTmax),"pdf");
  cDecayLengthNormalised->Print(Form("/home/dmorris/KaKa_%i_to_%i_pt.pdf",pTmin,pTmax),"pdf");
  cTrackDcaXY->Print(Form("/home/dmorris/KaKa_%i_to_%i_pt.pdf",pTmin,pTmax),"pdf");
  cCpa->Print(Form("/home/dmorris/KaKa_%i_to_%i_pt.pdf",pTmin,pTmax),"pdf");
  cCpaXY->Print(Form("/home/dmorris/KaKa_%i_to_%i_pt.pdf)",pTmin,pTmax),"pdf");

  ofile->Write();

cout<<"ciao"<<endl;

}
