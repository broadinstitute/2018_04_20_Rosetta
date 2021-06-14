# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
# score(X, y, sample_weight=None)[source]
# Return the coefficient of determination R^2 of the prediction.

# The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0.

from sklearn.model_selection import cross_val_score, cross_val_predict, GroupKFold,LeaveOneGroupOut
from sklearn import metrics
import numpy as np


def lasso_cv(X,y,k,group_labels):
    #####
    ## X: CP data [perts/samples, features]
    ## y: lm gene expression value [perts/samples, 1 (feature value)]
    from sklearn import linear_model
    n_j=3
    # build sklearn model
    clf = linear_model.Lasso(alpha=0.1,max_iter=10000)

#     k=np.unique(group_labels).shape[0]
    split_obj=GroupKFold(n_splits=k)
#     split_obj = LeaveOneGroupOut()    
    # Perform k-fold cross validation
    scores = cross_val_score(clf, X, y, groups=group_labels,cv=split_obj,n_jobs=n_j)
    
    
    
    # Perform k-fold cross validation on the shuffled vector of lm GE across samples
    # y.sample(frac = 1) this just shuffles the vector
    scores_rand = cross_val_score(clf, X, y.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
    return scores, scores_rand


def MLP_cv(X,y,k,group_labels):
    from sklearn.neural_network import MLPRegressor

    n_j=-1
#     hidden_layer_sizes=100,
#     hidden_layer_sizes = (50, 20, 10)
    regr = MLPRegressor(random_state=1,hidden_layer_sizes = (100), max_iter=10000,activation='tanh')

    split_obj=GroupKFold(n_splits=k)    
    # Perform k-fold cross validation
    scores = cross_val_score(regr, X, y, groups=group_labels,cv=split_obj,n_jobs=n_j)
    
    # Perform k-fold cross validation on the shuffled vector of lm GE across samples
    # y.sample(frac = 1) this just shuffles the vector
    scores_rand = cross_val_score(regr, X, y.sample(frac = 1) ,groups=group_labels,cv=split_obj,n_jobs=n_j)
    return scores, scores_rand






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
# datasets=['LUAD', 'TAORF', 'LINCS', 'CDRP-bio'];
# DT_kfold={'LUAD':10, 'TAORF':5, 'LINCS':20, 'CDRP-bio':20}
# # CP Profile Type options: 'augmented' , 'normalized', 'normalized_variable_selected'
# # 'normalized_feature_select_dmso'
# profileType='normalized'
# profileLevel='treatment'; #'replicate'  or  'treatment'
# highRepOverlapEnabled=0

# if highRepOverlapEnabled:
#     f='-filt'
# else:
#     f=''

# for dataset in datasets:
#     # n of samples for replicate picking options: numbers or, 'max'
    
#     if dataset=='LINCS':
# #         profileType='normalized_feature_select_dmso'
#         profileType="normalized_dmso"
#     else:
# #         profileType='normalized_variable_selected'      
#         profileType='normalized'       
    
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


#     scaler_ge = preprocessing.StandardScaler()
#     scaler_cp = preprocessing.StandardScaler()
#     l1k_scaled=l1k.copy()
#     l1k_scaled[l1k_features] = scaler_ge.fit_transform(l1k[l1k_features].values)
#     cp_scaled=cp.copy()
#     cp_scaled[cp_features] = scaler_cp.fit_transform(cp[cp_features].values.astype('float64'))

#     for model in ["Lasso"]:    
    
#         if model=="MLP":
#             cp_scaled[cp_features] =preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(cp_scaled[cp_features].values)   
#             l1k_scaled[l1k_features] =preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(l1k_scaled[l1k_features].values)           


#         if 1:
#             cp=cp_scaled.copy()
#             l1k=l1k_scaled.copy()

#         ##############################
         
             
            
            
#         k_fold=DT_kfold[dataset]
#         pred_df=pd.DataFrame(index=range(k_fold),columns=cp_features)
#         pred_df_rand=pd.DataFrame(index=range(k_fold),columns=cp_features)

#         for c in cp_features:
# #         for l in l1k_features:
#             if model=="Lasso":
#                 scores,scores_rand=lasso_cv(l1k[l1k_features],cp[c],k_fold)
# #                 scores,scores_rand=lasso_cv(cp[cp_features],l1k[l],k_fold)
#             elif model=="MLP":
#                 scores,scores_rand=MLP_cv(l1k[l1k_features],cp[c],k_fold)
# #                 scores,scores_rand=MLP_cv(cp[cp_features],l1k[l],k_fold)            
#             pred_df[c]=scores
#             pred_df_rand[c]=scores_rand

#         ########################### mapping prob_ids to genes names    
# #         meta=pd.read_csv("/home/ubuntu/bucket/projects/2018_04_20_Rosetta/workspace/metadata/affy_probe_gene_mapping.txt",delimiter="\t",header=None, names=["probe_id", "gene"])
# #         meta_gene_probID=meta.set_index('probe_id')
# #         d = dict(zip(meta_gene_probID.index, meta_gene_probID['gene']))
# #         pred_df = pred_df.rename(columns=d)    
# #         pred_df_rand = pred_df_rand.rename(columns=d)  


#         meltedPredDF=pd.melt(pred_df).rename(columns={'variable':'CP-Features','value':'pred score'})
#         meltedPredDF_rand=pd.melt(pred_df_rand).rename(columns={'variable':'CP-Features','value':'pred score'})
#         meltedPredDF['d']="n-folds"
#         meltedPredDF_rand['d']="random"
#         filename='../../results/SingleCPfeatPred/scores.xlsx'
#         saveAsNewSheetToExistingFile(filename,pd.concat([meltedPredDF,meltedPredDF_rand],ignore_index=True),\
#                                      model+'-'+dataset+'-dists'+f+'-paper')