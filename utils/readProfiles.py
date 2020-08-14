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

    

    ############## LUAD
    if dataset=='LUAD':
        labelCol='Allele'
        cp_data_repLevel=cp_data_repLevel.rename(columns={'x_mutation_status':labelCol})
        l1k_data_repLevel=l1k_data_repLevel.rename(columns={'allele':labelCol})
        
        cp_features=cp_data_repLevel.columns[cp_data_repLevel.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")]
        l1k_features=l1k_data_repLevel.columns[l1k_data_repLevel.columns.str.contains("_at")]

        l1k_data_treatLevel=l1k_data_repLevel.groupby(labelCol)[l1k_features].mean().reset_index();
        cp_data_treatLevel=cp_data_repLevel.groupby(labelCol)[cp_features].mean().reset_index();
        
        print('Replicate Level Shapes (nSamples x nFeatures): cp: ',\
              cp_data_repLevel.shape[0],',',len(cp_features),  ',  l1k: ',l1k_data_repLevel.shape[0],',',len(l1k_features))
        
        print('l1k n of rep: ',l1k_data_repLevel.groupby([labelCol]).size().median())
        print('cp n of rep: ',cp_data_repLevel.groupby([labelCol]).size().median())
        meta_cp=cp_data_repLevel[['Allele','Metadata_broad_sample_type','Metadata_pert_type']].drop_duplicates().reset_index(drop=True)
    #     meta_l1k=l1k_data_repLevel[['Metadata_pert_id','CPD_NAME','CPD_TYPE','CPD_SMILES']].drop_duplicates().reset_index(drop=True)

    # #     l1k_data_treatLevel=pd.merge(l1k_data_treatLevel,meta_l1k, how='inner',on=['allele'])
        cp_data_treatLevel=pd.merge(cp_data_treatLevel,meta_cp, how='inner',on=[labelCol])

        mergedProfiles_treatLevel=pd.merge(cp_data_treatLevel, l1k_data_treatLevel, how='inner',on=[labelCol])

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
        print('Replicate Level Shapes (nSamples x nFeatures): cp: ',
              cp_data_repLevel.shape[0],',',len(cp_features),  ',  l1k: ',l1k_data_repLevel.shape[0],',',len(l1k_features))
        
        print('l1k n of rep: ',l1k_data_repLevel.groupby(['Metadata_pert_id']).size().median())
        print('cp n of rep: ',cp_data_repLevel.groupby(['Metadata_pert_id']).size().median())
        
        meta_cp=cp_data_repLevel[['Metadata_pert_id','Metadata_moa','Metadata_target']].drop_duplicates().reset_index(drop=True)
        meta_l1k=l1k_data_repLevel[['Metadata_pert_id','CPD_NAME','CPD_TYPE','CPD_SMILES']].drop_duplicates().reset_index(drop=True)

        l1k_data_treatLevel=pd.merge(l1k_data_treatLevel,meta_l1k, how='inner',on=['Metadata_pert_id'])
        cp_data_treatLevel=pd.merge(cp_data_treatLevel,meta_cp, how='inner',on=['Metadata_pert_id'])

        mergedProfiles_treatLevel=pd.merge(cp_data_treatLevel, l1k_data_treatLevel, how='inner',on=['Metadata_pert_id'])

        print('Treatment Level Shapes (nSamples x nFeatures):',cp_data_treatLevel.shape,l1k_data_treatLevel.shape,\
              'Merged Profiles Shape:', mergedProfiles_treatLevel.shape)

    ############### TA ORF
    if dataset=='TAORF':
#         labelCol='Metadata_gene_name'
        labelCol='Metadata_broad_sample'
        l1k_data_repLevel=l1k_data_repLevel.rename(columns={'pert_id':labelCol})

        cp_features=cp_data_repLevel.columns[cp_data_repLevel.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")]
        l1k_features=l1k_data_repLevel.columns[l1k_data_repLevel.columns.str.contains("_at")]

        l1k_data_treatLevel=l1k_data_repLevel.groupby(labelCol)[l1k_features].mean().reset_index();
        cp_data_treatLevel=cp_data_repLevel.groupby(labelCol)[cp_features].mean().reset_index();
        
        print('Replicate Level Shapes (nSamples x nFeatures): cp: ',
              cp_data_repLevel.shape[0],',',len(cp_features),  ',  l1k: ',l1k_data_repLevel.shape[0],',',len(l1k_features))
#         meta_cp=cp_data_repLevel[['Metadata_gene_name','Metadata_moa','Allele']].drop_duplicates().reset_index(drop=True)
        meta_cp=cp_data_repLevel[[labelCol,'Metadata_moa']].drop_duplicates().reset_index(drop=True)
        meta_l1k=l1k_data_repLevel[[labelCol,'pert_type']].drop_duplicates().reset_index(drop=True)
        print('l1k n of rep: ',l1k_data_repLevel.groupby([labelCol]).size().median())
        print('cp n of rep: ',cp_data_repLevel.groupby([labelCol]).size().median()) 
        
        l1k_data_treatLevel=pd.merge(l1k_data_treatLevel,meta_l1k, how='inner',on=[labelCol])
        cp_data_treatLevel=pd.merge(cp_data_treatLevel,meta_cp, how='inner',on=[labelCol])

        mergedProfiles_treatLevel=pd.merge(cp_data_treatLevel, l1k_data_treatLevel, how='inner',on=[labelCol]).drop_duplicates().reset_index(drop=True)

        print('Treatment Level Shapes (nSamples x nFeatures+metadata):',cp_data_treatLevel.shape,l1k_data_treatLevel.shape,\
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


