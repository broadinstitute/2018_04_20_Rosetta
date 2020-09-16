import numpy as np
import scipy.spatial
import pandas as pd
import sklearn.decomposition
from sklearn import preprocessing
from sklearn.metrics import pairwise_distances
from utils.normalize_funcs import standardize_per_catX

# def readMergedProfiles(dataset,profileType,nRep):
def readMergedProfiles(dataset_rootDir,dataset,profileType,profileLevel,nRep):

    #'dataset_name',['folder_name',[cp_pert_col_name,l1k_pert_col_name],[cp_control_val,l1k_control_val]]
    ds_info_dict={'CDRP':['CDRPBIO-BBBC036-Bray',['Metadata_broad_sample','pert_id_dose'],[['DMSO_0.04'],['DMSO_-666']]],
                  'TAORF':['TA-ORF-BBBC037-Rohban',['Metadata_broad_sample','pert_id',],[['DMSO_0.04'],['DMSO_-666']]],
                  'LUAD':['LUAD-BBBC041-Caicedo',['x_mutation_status','allele'],[['DMSO_0.04'],['DMSO_-666']]],
                  'LINCS':['LINCS-Pilot1',['Metadata_pert_id_dose','pert_id_dose'],[[np.nan],['DMSO_-666']]]}
    
    dataDir=dataset_rootDir+'/preprocessed_data/'+ds_info_dict[dataset][0]+'/'
        
    cp_data_repLevel=pd.read_csv(dataDir+'/CellPainting/replicate_level_cp_'+profileType+'.csv.gz')    
    l1k_data_repLevel=pd.read_csv(dataDir+'/L1000/replicate_level_l1k.csv.gz')  

    
    # features to analyse
    cp_features=cp_data_repLevel.columns[cp_data_repLevel.columns.str.contains("Cells_|Cytoplasm_|Nuclei_")].tolist()
    l1k_features=l1k_data_repLevel.columns[l1k_data_repLevel.columns.str.contains("_at")].tolist()
    
    
    
    ########## removes nans and infs
    l1k_data_repLevel=l1k_data_repLevel.replace([np.inf, -np.inf], np.nan)
    cp_data_repLevel=cp_data_repLevel.replace([np.inf, -np.inf], np.nan)
    cols2remove0=[i for i in cp_features if (cp_data_repLevel[i].isnull().sum(axis=0)/cp_data_repLevel.shape[0])>0.05]
    # cols2removeCP=[i for i in cp.columns.tolist() if cp[i].isnull().sum(axis=0)>0]
#     print(cols2removeCP)
#     print(len(cp_features))

#     cols2remove0=[i for i in cpFeatures if ((pd_df[i]=='nan').sum(axis=0)/pd_df.shape[0])>0.05]
    print(cols2remove0)
    
#     cols2remove1=cpFeatures[pd_df[cpFeatures].std().values<0.00001].tolist()
    cols2remove1=cp_data_repLevel[cp_features].std()[cp_data_repLevel[cp_features].std() < 0.0001].index.tolist()
    print(cols2remove1)    
    cols2removeCP=cols2remove0+cols2remove1
#     print(cols2removeCP)

    cp_features = list(set(cp_features) - set(cols2removeCP))
#     print(len(cp_features))
    cp_data_repLevel=cp_data_repLevel.drop(cols2removeCP, axis=1);
    cp_data_repLevel[cp_features] = cp_data_repLevel[cp_features].interpolate()
    
    cols2removeCP=[i for i in cp_features if cp_data_repLevel[i].isnull().sum(axis=0)>0]
    print(cols2removeCP)
    
#     cp=cp.fillna(cp.median())

    # cols2removeGE=[i for i in l1k.columns if l1k[i].isnull().sum(axis=0)>0]
    # print(cols2removeGE)
    # l1k_features = list(set(l1k_features) - set(cols2removeGE))
    # print(len(l1k_features))
    # l1k=l1k.drop(cols2removeGE, axis=1);
    l1k_data_repLevel[l1k_features] = l1k_data_repLevel[l1k_features].interpolate()
    # l1k=l1k.fillna(l1k.median())
    
    
    
    ################ Per plate scaling 
    cp_data_repLevel = standardize_per_catX(cp_data_repLevel,'Metadata_Plate',cp_features);
    l1k_data_repLevel = standardize_per_catX(l1k_data_repLevel,'det_plate',l1k_features);    
    cols2removeCP=[i for i in cp_features if (cp_data_repLevel[i].isnull().sum(axis=0)/cp_data_repLevel.shape[0])>0.05]
    cp_data_repLevel=cp_data_repLevel.drop(cols2removeCP, axis=1);
    cp_features = list(set(cp_features) - set(cols2removeCP))
    cp_data_repLevel[cp_features] = cp_data_repLevel[cp_features].interpolate()
    
    # rename columns that should match
    labelCol='PERT'
#     print(cp_data_repLevel[ds_info_dict[dataset][1][0]])
    cp_data_repLevel=cp_data_repLevel.rename(columns={ds_info_dict[dataset][1][0]:labelCol})
    l1k_data_repLevel=l1k_data_repLevel.rename(columns={ds_info_dict[dataset][1][1]:labelCol})    
            
    
    ###### print some data statistics
    print(dataset+': Replicate Level Shapes (nSamples x nFeatures): cp: ',\
          cp_data_repLevel.shape[0],',',len(cp_features),  ',  l1k: ',l1k_data_repLevel.shape[0],',',len(l1k_features))

    print('l1k n of rep: ',l1k_data_repLevel.groupby([labelCol]).size().median())
    print('cp n of rep: ',cp_data_repLevel.groupby([labelCol]).size().median())
    

    ###### remove perts with low rep corr
    if 1:
        highRepPerts = highRepFinder(dataset) + ['DMSO'];
        cp_data_repLevel=cp_data_repLevel[cp_data_repLevel['PERT'].isin(highRepPerts)].reset_index()
        l1k_data_repLevel=l1k_data_repLevel[l1k_data_repLevel['PERT'].isin(highRepPerts)].reset_index()        
    
    ####### form treatment level profiles
    l1k_data_treatLevel=l1k_data_repLevel.groupby(labelCol)[l1k_features].mean().reset_index();
    cp_data_treatLevel=cp_data_repLevel.groupby(labelCol)[cp_features].mean().reset_index();
    
    cols2removeCP=[i for i in cp_features if cp_data_treatLevel[i].isnull().sum(axis=0)>0]
    print(cols2removeCP)
    
    ###### define metadata and merge treatment level profiles
#     dataset:[[cp_columns],[l1k_columns]]
    meta_dict={'CDRP':[['Metadata_moa','Metadata_target'],['CPD_NAME','CPD_TYPE','CPD_SMILES']],
              'TAORF':[['Metadata_moa'],['pert_type']],
              'LUAD':[['Metadata_broad_sample_type','Metadata_pert_type'],[]],
              'LINCS':[[],[]]}
    
    
    meta_cp=cp_data_repLevel[[labelCol]+meta_dict[dataset][0]].\
    drop_duplicates().reset_index(drop=True)
    meta_l1k=l1k_data_repLevel[[labelCol]+meta_dict[dataset][1]].\
    drop_duplicates().reset_index(drop=True)

    # #     l1k_data_treatLevel=pd.merge(l1k_data_treatLevel,meta_l1k, how='inner',on=['allele'])
    cp_data_treatLevel=pd.merge(cp_data_treatLevel,meta_cp, how='inner',on=[labelCol])
    l1k_data_treatLevel=pd.merge(l1k_data_treatLevel,meta_l1k, how='inner',on=[labelCol])

    mergedProfiles_treatLevel=pd.merge(cp_data_treatLevel, l1k_data_treatLevel, how='inner',on=[labelCol])

    print('Treatment Level Shapes (nSamples x nFeatures+metadata):',cp_data_treatLevel.shape,l1k_data_treatLevel.shape,\
          'Merged Profiles Shape:', mergedProfiles_treatLevel.shape)

    
    if profileLevel=='replicate':
        ## calculate rep level
        if nRep=='max':
            cp_data_n_repLevel=cp_data_repLevel.copy()
            l1k_data_n_repLevel=l1k_data_repLevel.copy()
        else:
    #         nR=np.min((cp_data_repLevel.groupby(labelCol).size().min(),l1k_data_repLevel.groupby(labelCol).size().min()))
    #     cp_data_n_repLevel=cp_data_repLevel.groupby(labelCol).apply(lambda x: x.sample(n=nR,replace=True)).reset_index(drop=True)
            nR=nRep
            cp_data_n_repLevel=cp_data_repLevel.groupby(labelCol).\
            apply(lambda x: x.sample(n=np.min([nR,x.shape[0]]))).reset_index(drop=True)
            l1k_data_n_repLevel=l1k_data_repLevel.groupby(labelCol).\
            apply(lambda x: x.sample(n=np.min([nR,x.shape[0]]))).reset_index(drop=True)


        mergedProfiles_repLevel=pd.merge(cp_data_n_repLevel, l1k_data_n_repLevel, how='inner',on=[labelCol])
    else:
        mergedProfiles_repLevel=[]
    
    
    
    return mergedProfiles_repLevel,mergedProfiles_treatLevel,cp_features,l1k_features



########################### old func
def readMergedProfiles2(dataset,profileType,nRep):

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



def highRepFinder(dataset):
    repCorDF=pd.read_excel('../../results/RepCor/RepCorrDF.xlsx', sheet_name=None)
    cpRepDF=repCorDF['cp-'+dataset.lower()]
    cpHighList=cpRepDF[cpRepDF['RepCor']>cpRepDF['Rand90Perc']]['Unnamed: 0'].tolist()
    print('CP: from ',cpRepDF.shape[0],' to ',len(cpHighList))
    cpRepDF=repCorDF['l1k-'+dataset.lower()]
    l1kHighList=cpRepDF[cpRepDF['RepCor']>cpRepDF['Rand90Perc']]['Unnamed: 0'].tolist()
    highRepPerts=list(set(l1kHighList) & set(cpHighList))
    print('l1k: from ',cpRepDF.shape[0],' to ',len(l1kHighList))
    print('CP and l1k high rep overlap: ',len(highRepPerts))
    return highRepPerts