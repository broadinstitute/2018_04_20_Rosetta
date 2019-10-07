import numpy as np
import scipy.spatial
import pandas as pd
import sklearn.decomposition
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances


def readMergedProfiles(dataset,profileType,nRep):

    if dataset=='CDRP':
        dataDir='~/workspace_rosetta/workspace/preprocessed_data/CDRPBIO-BBBC036-Bray/'
    elif dataset=='TAORF':
        dataDir='~/workspace_rosetta/workspace/preprocessed_data/TA-ORF-BBBC037-Rohban/'
    elif dataset=='LUAD':
        dataDir='~/workspace_rosetta/workspace/preprocessed_data/LUAD-BBBC041-Caicedo/'

    cp_data_repLevel=pd.read_csv(dataDir+'/CellPainting/replicate_level_cp_'+profileType+'.csv')    
    l1k_data_repLevel=pd.read_csv(dataDir+'/L1000/replicate_level_l1k.csv')  

    print('Replicate Level Shapes (nSamples x nFeatures): cp: ',cp_data_repLevel.shape,'  l1k: ',l1k_data_repLevel.shape)

    ############## LUAD
    if dataset=='LUAD':
        labelCol='Allele'
        cp_data_repLevel=cp_data_repLevel.rename(columns={'x_mutation_status':'Allele'})
        l1k_data_repLevel=l1k_data_repLevel.rename(columns={'allele':'Allele'})
        
        cp_features=cp_data_repLevel.columns[cp_data_repLevel.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")]
        l1k_features=l1k_data_repLevel.columns[l1k_data_repLevel.columns.str.contains("_at")]

        l1k_data_treatLevel=l1k_data_repLevel.groupby('Allele')[l1k_features].mean().reset_index();
        cp_data_treatLevel=cp_data_repLevel.groupby('Allele')[cp_features].mean().reset_index();

        meta_cp=cp_data_repLevel[['Allele','Metadata_broad_sample_type','Metadata_pert_type']].drop_duplicates().reset_index(drop=True)
    #     meta_l1k=l1k_data_repLevel[['Metadata_pert_id','CPD_NAME','CPD_TYPE','CPD_SMILES']].drop_duplicates().reset_index(drop=True)

    # #     l1k_data_treatLevel=pd.merge(l1k_data_treatLevel,meta_l1k, how='inner',on=['allele'])
        cp_data_treatLevel=pd.merge(cp_data_treatLevel,meta_cp, how='inner',on=['Allele'])

        mergedProfiles_treatLevel=pd.merge(cp_data_treatLevel, l1k_data_treatLevel, how='inner',on=['Allele'])

        print('Treatment Level Shapes (nSamples x nFeatures):',cp_data_treatLevel.shape,l1k_data_treatLevel.shape,\
              'Merged Profiles Shape:', mergedProfiles_treatLevel.shape)


    
    ############## CDRP
    if dataset=='CDRP':
        labelCol='Metadata_pert_id'
        l1k_data_repLevel=l1k_data_repLevel.rename(columns={'BROAD_CPD_ID':'Metadata_pert_id'})

        cp_features=cp_data_repLevel.columns[cp_data_repLevel.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")]
        l1k_features=l1k_data_repLevel.columns[l1k_data_repLevel.columns.str.contains("_at")]

        l1k_data_treatLevel=l1k_data_repLevel.groupby('Metadata_pert_id')[l1k_features].mean().reset_index();
        cp_data_treatLevel=cp_data_repLevel.groupby('Metadata_pert_id')[cp_features].mean().reset_index();

        meta_cp=cp_data_repLevel[['Metadata_pert_id','Metadata_moa','Metadata_target']].drop_duplicates().reset_index(drop=True)
        meta_l1k=l1k_data_repLevel[['Metadata_pert_id','CPD_NAME','CPD_TYPE','CPD_SMILES']].drop_duplicates().reset_index(drop=True)

        l1k_data_treatLevel=pd.merge(l1k_data_treatLevel,meta_l1k, how='inner',on=['Metadata_pert_id'])
        cp_data_treatLevel=pd.merge(cp_data_treatLevel,meta_cp, how='inner',on=['Metadata_pert_id'])

        mergedProfiles_treatLevel=pd.merge(cp_data_treatLevel, l1k_data_treatLevel, how='inner',on=['Metadata_pert_id'])

        print('Treatment Level Shapes (nSamples x nFeatures):',cp_data_treatLevel.shape,l1k_data_treatLevel.shape,\
              'Merged Profiles Shape:', mergedProfiles_treatLevel.shape)

    ############### TA ORF
    if dataset=='TAORF':
        labelCol='Metadata_gene_name'
        l1k_data_repLevel=l1k_data_repLevel.rename(columns={'x_genesymbol_mutation':'Metadata_gene_name'})

        cp_features=cp_data_repLevel.columns[cp_data_repLevel.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")]
        l1k_features=l1k_data_repLevel.columns[l1k_data_repLevel.columns.str.contains("_at")]

        l1k_data_treatLevel=l1k_data_repLevel.groupby('Metadata_gene_name')[l1k_features].mean().reset_index();
        cp_data_treatLevel=cp_data_repLevel.groupby('Metadata_gene_name')[cp_features].mean().reset_index();

        meta_cp=cp_data_repLevel[['Metadata_gene_name','Metadata_pert_name','Metadata_moa','Allele','Metadata_broad_sample_2']].drop_duplicates().reset_index(drop=True)
        meta_l1k=l1k_data_repLevel[['Metadata_gene_name','pert_type']].drop_duplicates().reset_index(drop=True)

        l1k_data_treatLevel=pd.merge(l1k_data_treatLevel,meta_l1k, how='inner',on=['Metadata_gene_name'])
        cp_data_treatLevel=pd.merge(cp_data_treatLevel,meta_cp, how='inner',on=['Metadata_gene_name'])

        mergedProfiles_treatLevel=pd.merge(cp_data_treatLevel, l1k_data_treatLevel, how='inner',on=['Metadata_gene_name'])

        print('Treatment Level Shapes (nSamples x nFeatures):',cp_data_treatLevel.shape,l1k_data_treatLevel.shape,\
              'Merged Profiles Shape:', mergedProfiles_treatLevel.shape)
    
    ## calculate rep level
    if nRep=='max':
        cp_data_n_repLevel=cp_data_repLevel.copy()
        l1k_data_n_repLevel=l1k_data_repLevel.copy()
    else:
#         nR=np.min((cp_data_repLevel.groupby(labelCol).size().min(),l1k_data_repLevel.groupby(labelCol).size().min()))
#     cp_data_n_repLevel=cp_data_repLevel.groupby(labelCol).apply(lambda x: x.sample(n=nR,replace=True)).reset_index(drop=True)
        nR=nRep
        cp_data_n_repLevel=cp_data_repLevel.groupby(labelCol).apply(lambda x: x.sample(n=np.min([nR,x.shape[0]]))).reset_index(drop=True)
        l1k_data_n_repLevel=l1k_data_repLevel.groupby(labelCol).apply(lambda x: x.sample(n=np.min([nR,x.shape[0]]))).reset_index(drop=True)


    mergedProfiles_repLevel=pd.merge(cp_data_n_repLevel, l1k_data_n_repLevel, how='inner',on=[labelCol])
    
    
    
    return mergedProfiles_repLevel,mergedProfiles_treatLevel,cp_features,l1k_features,labelCol


# input is a list of dfs--> [cp,l1k,cp_cca,l1k_cca]
#######
def plotRepCorrs(allData,pertName):
    corrAll=[]
    for d in range(len(allData)):
        df=allData[d];
        uniqPert=df[pertName].unique().tolist()
        repC=[]
        randC=[]
        for u in uniqPert:
            df1=df[df[pertName]==u].drop_duplicates().reset_index(drop=True)
            df2=df[df[pertName]!=u].drop_duplicates().reset_index(drop=True)
            repCorr=np.sort(np.unique(df1.iloc[:,1:].T.corr().values))[:-1].tolist()
            repC=repC+repCorr
            randAllels=df2[pertName].drop_duplicates().sample(df1.shape[0]).tolist()
            df3=pd.concat([df2[df2[pertName]==i].reset_index(drop=True).iloc[0:1,:] for i in randAllels],ignore_index=True)
            randCorr=df1.corrwith(df3, axis = 1,method='pearson').values.tolist()
            randC=randC+randCorr
        
        corrAll.append([randC,repC]);
    return corrAll