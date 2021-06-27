# %matplotlib inline
# %load_ext autoreload
# %autoreload 2
# from comet_ml import Experiment
import numpy as np
import scipy.spatial
import pandas as pd
import comet_ml
import sklearn.decomposition
import matplotlib.pyplot as plt
# import keras
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances,mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from utils.readProfiles import readMergedProfiles,readMergedProfiles2
from utils.pred_models import *
from utils.saveAsNewSheetToExistingFile import saveAsNewSheetToExistingFile

sns.set_style("whitegrid")
# from utils import networksEvol, tsne, readProfiles
import umap


# meta_geneFamily = pd.read_csv("/home/ubuntu/bucket/projects/2018_04_20_Rosetta/workspace/metadata/hgnc_gene_group_family_dictionary.csv")
# genesU = meta_geneFamily.approved_symbol.unique().tolist()
# meta_geneFamilyU=pd.DataFrame(index=range(len(genesU)),columns=["lmGens","gene_group_name"])
# meta_geneFamilyU["lmGens"]=genesU
# for i in range(len(genesU)):
#     meta_geneFamilyU.loc[meta_geneFamilyU["lmGens"]==genesU[i],"gene_group_name"]=', '.join(meta_geneFamily[meta_geneFamily["approved_symbol"]==genesU[i]]["gene_group_name"].tolist())
# # meta_geneFamilyU


# dataset_rootDir='./';pertColName='PERT'
# # dataset options: 'CDRP' , 'LUAD', 'TAORF', 'LINCS', 'CDRP-bio'
# # datasets=['LUAD', 'TAORF', 'LINCS', 'CDRP-bio'];
# datasets=['LUAD', 'LINCS', 'CDRP-bio'];
# DT_kfold={'LUAD':10, 'TAORF':5, 'LINCS':20, 'CDRP-bio':20}
# # CP Profile Type options: 'augmented' , 'normalized', 'normalized_variable_selected'
# # 'normalized_feature_select_dmso'
# profileType='normalized'
# profileLevel='treatment'; #'replicate'  or  'treatment'
# highRepOverlapEnabled=1

# if highRepOverlapEnabled:
#     f='-filt'
# else:
#     f=''

# for dataset in datasets:
#     # n of samples for replicate picking options: numbers or, 'max'
    
#     if dataset=='LINCS':
#         profileType='normalized_feature_select_dmso'
# #         profileType="normalized_dmso"
#     else:
#         profileType='normalized_variable_selected'      
# #         profileType='normalized'       
    
#     nRep=2
#     mergProf_repLevel,mergProf_treatLevel,cp_features,l1k_features=\
#     readMergedProfiles(dataset_rootDir,dataset,profileType,profileLevel,nRep,highRepOverlapEnabled);
#     # mergProf_repLevel,mergProf_treatLevel,l1k_features,cp_features,pertColName=readMergedProfiles(dataset,profileType,nRep)
#     # cp_features,l1k_features=cp_features.tolist(),l1k_features.tolist()


#     if profileLevel=='replicate':
#         l1k=mergProf_repLevel[[pertColName]+l1k_features]
#         cp=mergProf_repLevel[[pertColName]+cp_features]
#     elif profileLevel=='treatment':
#         l1k=mergProf_treatLevel[[pertColName]+l1k_features]
#         cp=mergProf_treatLevel[[pertColName]+cp_features]
        
        
#         if dataset=='LINCS':     
#             cp['Compounds']=cp['PERT'].str[0:13]
#             l1k['Compounds']=l1k['PERT'].str[0:13]
#         else:
#             cp['Compounds']=cp['PERT']
#             l1k['Compounds']=l1k['PERT']      
            
            
#         le = preprocessing.LabelEncoder()
#         group_labels=le.fit_transform(l1k['Compounds'].values)

     

#     scaler_ge = preprocessing.StandardScaler()
#     scaler_cp = preprocessing.StandardScaler()
#     l1k_scaled=l1k.copy()
#     l1k_scaled[l1k_features] = scaler_ge.fit_transform(l1k[l1k_features].values)
#     cp_scaled=cp.copy()
#     cp_scaled[cp_features] = scaler_cp.fit_transform(cp[cp_features].values.astype('float64'))

    
    
    
#     for model in ["MLP"]:    
    
#         if model=="MLP":
#             cp_scaled[cp_features] =preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(cp_scaled[cp_features].values)   
#             l1k_scaled[l1k_features] =preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(l1k_scaled[l1k_features].values)           


#         if 1:
#             cp=cp_scaled.copy()
#             l1k=l1k_scaled.copy()

#         ##############################
         
# #         k_fold=DT_kfold[dataset]
# #         k_fold=int(np.unique(group_labels).shape[0]/4)
#         k_fold=int(np.unique(group_labels).shape[0]/20)    
#         pred_df=pd.DataFrame(index=range(k_fold),columns=l1k_features)
#         pred_df_rand=pd.DataFrame(index=range(k_fold),columns=l1k_features)
#         for l in l1k_features:
#             if model=="Lasso":
#                 scores,scores_rand=lasso_cv(cp[cp_features],l1k[l],k_fold,group_labels)
#             elif model=="MLP":
#                 scores,scores_rand=MLP_cv(cp[cp_features],l1k[l],k_fold,group_labels)            
#             pred_df[l]=scores
#             pred_df_rand[l]=scores_rand

#         ########################### mapping prob_ids to genes names    
#         meta=pd.read_csv("/home/ubuntu/bucket/projects/2018_04_20_Rosetta/workspace/metadata/affy_probe_gene_mapping.txt",delimiter="\t",header=None, names=["probe_id", "gene"])
#         meta_gene_probID=meta.set_index('probe_id')
#         d = dict(zip(meta_gene_probID.index, meta_gene_probID['gene']))
#         pred_df = pred_df.rename(columns=d)    
#         pred_df_rand = pred_df_rand.rename(columns=d)  


#         meltedPredDF=pd.melt(pred_df).rename(columns={'variable':'lmGens','value':'pred score'})
#         meltedPredDF_rand=pd.melt(pred_df_rand).rename(columns={'variable':'lmGens','value':'pred score'})
#         meltedPredDF['d']="n-folds"
#         meltedPredDF_rand['d']="random"
#         filename='../../results/SingleGenePred/scores_group.xlsx'
#         saveAsNewSheetToExistingFile(filename,pd.concat([meltedPredDF,meltedPredDF_rand],ignore_index=True),\
#                                      model+'-'+dataset+'-fSel-dists'+f+'-kG2')
        
        
 


  ########################### for prediction of cp features
    
meta_geneFamily = pd.read_csv("/home/ubuntu/bucket/projects/2018_04_20_Rosetta/workspace/metadata/hgnc_gene_group_family_dictionary.csv")
genesU = meta_geneFamily.approved_symbol.unique().tolist()
meta_geneFamilyU=pd.DataFrame(index=range(len(genesU)),columns=["lmGens","gene_group_name"])
meta_geneFamilyU["lmGens"]=genesU
for i in range(len(genesU)):
    meta_geneFamilyU.loc[meta_geneFamilyU["lmGens"]==genesU[i],"gene_group_name"]=', '.join(meta_geneFamily[meta_geneFamily["approved_symbol"]==genesU[i]]["gene_group_name"].tolist())
# meta_geneFamilyU


dataset_rootDir='./';pertColName='PERT'
# dataset options: 'CDRP' , 'LUAD', 'TAORF', 'LINCS', 'CDRP-bio'
# datasets=['LUAD', 'TAORF', 'LINCS', 'CDRP-bio'];
datasets=['CDRP-bio'];
DT_kfold={'LUAD':10, 'TAORF':5, 'LINCS':20, 'CDRP-bio':20}
# CP Profile Type options: 'augmented' , 'normalized', 'normalized_variable_selected'
# 'normalized_feature_select_dmso'
profileType='normalized'
profileLevel='treatment'; #'replicate'  or  'treatment'
highRepOverlapEnabled=1

if highRepOverlapEnabled:
    f='-filt'
else:
    f=''

for dataset in datasets:
    # n of samples for replicate picking options: numbers or, 'max'
    
    if dataset=='LINCS':
#         profileType='normalized_feature_select_dmso'
        profileType="normalized_dmso"
    else:
#         profileType='normalized_variable_selected'      
        profileType='normalized'       
    
    nRep=2
    mergProf_repLevel,mergProf_treatLevel,cp_features,l1k_features=\
    readMergedProfiles(dataset_rootDir,dataset,profileType,profileLevel,nRep,highRepOverlapEnabled);
    # mergProf_repLevel,mergProf_treatLevel,l1k_features,cp_features,pertColName=readMergedProfiles(dataset,profileType,nRep)
    # cp_features,l1k_features=cp_features.tolist(),l1k_features.tolist()


    if profileLevel=='replicate':
        l1k=mergProf_repLevel[[pertColName]+l1k_features]
        cp=mergProf_repLevel[[pertColName]+cp_features]
    elif profileLevel=='treatment':
        l1k=mergProf_treatLevel[[pertColName]+l1k_features]
        cp=mergProf_treatLevel[[pertColName]+cp_features]

        
    if dataset=='LINCS':     
        cp['Compounds']=cp['PERT'].str[0:13]
        l1k['Compounds']=l1k['PERT'].str[0:13]
    else:
        cp['Compounds']=cp['PERT']
        l1k['Compounds']=l1k['PERT']      


    le = preprocessing.LabelEncoder()
    group_labels=le.fit_transform(l1k['Compounds'].values)        
        

    scaler_ge = preprocessing.StandardScaler()
    scaler_cp = preprocessing.StandardScaler()
    l1k_scaled=l1k.copy()
    l1k_scaled[l1k_features] = scaler_ge.fit_transform(l1k[l1k_features].values)
    cp_scaled=cp.copy()
    cp_scaled[cp_features] = scaler_cp.fit_transform(cp[cp_features].values.astype('float64'))

    for model in ["Lasso"]:    
    
        if model=="MLP":
            cp_scaled[cp_features] =preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(cp_scaled[cp_features].values)   
            l1k_scaled[l1k_features] =preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(l1k_scaled[l1k_features].values)           


        if 1:
            cp=cp_scaled.copy()
            l1k=l1k_scaled.copy()

        ##############################
# #         k_fold=DT_kfold[dataset]
# #         k_fold=int(np.unique(group_labels).shape[0]/4)
#         k_fold=int(np.unique(group_labels).shape[0]/20)    
#         pred_df=pd.DataFrame(index=range(k_fold),columns=l1k_features)
#         pred_df_rand=pd.DataFrame(index=range(k_fold),columns=l1k_features)
#         for l in l1k_features:
#             if model=="Lasso":
#                 scores,scores_rand=lasso_cv(cp[cp_features],l1k[l],k_fold,group_labels)
#             elif model=="MLP":
#                 scores,scores_rand=MLP_cv(cp[cp_features],l1k[l],k_fold,group_labels)            
#             pred_df[l]=scores
#             pred_df_rand[l]=scores_rand         
             
            
            
        k_fold=DT_kfold[dataset]
        pred_df=pd.DataFrame(index=range(k_fold),columns=cp_features)
        pred_df_rand=pd.DataFrame(index=range(k_fold),columns=cp_features)

        for c in cp_features:
#         for l in l1k_features:
            if model=="Lasso":
                scores,scores_rand=lasso_cv(l1k[l1k_features],cp[c],k_fold,group_labels)
#                 scores,scores_rand=lasso_cv(cp[cp_features],l1k[l],k_fold)
            elif model=="MLP":
                scores,scores_rand=MLP_cv(l1k[l1k_features],cp[c],k_fold,group_labels)
#                 scores,scores_rand=MLP_cv(cp[cp_features],l1k[l],k_fold)            
            pred_df[c]=scores
            pred_df_rand[c]=scores_rand

        ########################### mapping prob_ids to genes names    
#         meta=pd.read_csv("/home/ubuntu/bucket/projects/2018_04_20_Rosetta/workspace/metadata/affy_probe_gene_mapping.txt",delimiter="\t",header=None, names=["probe_id", "gene"])
#         meta_gene_probID=meta.set_index('probe_id')
#         d = dict(zip(meta_gene_probID.index, meta_gene_probID['gene']))
#         pred_df = pred_df.rename(columns=d)    
#         pred_df_rand = pred_df_rand.rename(columns=d)  


        meltedPredDF=pd.melt(pred_df).rename(columns={'variable':'CP-Features','value':'pred score'})
        meltedPredDF_rand=pd.melt(pred_df_rand).rename(columns={'variable':'CP-Features','value':'pred score'})
        meltedPredDF['d']="n-folds"
        meltedPredDF_rand['d']="random"
        filename='../../results/SingleCPfeatPred/scores.xlsx'
        saveAsNewSheetToExistingFile(filename,pd.concat([meltedPredDF,meltedPredDF_rand],ignore_index=True),\
                                     model+'-'+dataset+'-dists'+f+'-paper')    
    
    
        