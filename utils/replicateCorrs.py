import numpy as np
import scipy.spatial
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from random import sample,choices
from scipy.stats import pearsonr
sb.set_style("whitegrid")

def replicateCorrs(inDf,pertColName,featColNames,plotEnabled):
    
    """ 
    Calculates replicate correlation versus across purtburtion correlations
  
    This function takes the input dataframe and output/plot replicate correlations. 
  
    Parameters: 
    inDf   (pandas df): input dataframe contains metadata and features
    pertColName  (str): The column based on which we define replicates of a purturbation
    featColNames(list): The column based on which we define replicates of a purturbation
    plotEnabled (bool): If True or 1, plots the curves 
    
    Returns: 
    int: Description of return value 
  
    """
    df=inDf.copy();
    uniqPert=df[pertColName].unique().tolist()
    repC=[]
    randC=[]
    for u in uniqPert:
        df1=df[df[pertColName]==u].drop_duplicates().reset_index(drop=True)
        df2=df[df[pertColName]!=u].drop_duplicates().reset_index(drop=True)

        repCorrPurtbs=df1.loc[:,featColNames].T.corr()
        repCorr=list(repCorrPurtbs.values[np.triu_indices(repCorrPurtbs.shape[0], k = 1)])
        
#         repCorr=np.sort(np.unique(df1.loc[:,featColNames].T.corr().values))[:-1].tolist()
        repC=repC+repCorr
        
#         randPertbs=df2[pertColName].drop_duplicates().sample(df1.shape[0],replace=True).tolist()
        nS=np.min([len(df2[pertColName].unique().tolist()),df1.shape[0]])
#         nS=df1.shape[0]
        
#         print(nS,[len(df2[pertColName].unique().tolist()),df1.shape[0]])
        
        randPertbs=sample(df2[pertColName].unique().tolist(),k=nS)
#         print(randPertbs)
        df3=pd.concat([df2[df2[pertColName]==i].sample(1,replace=True) for i in randPertbs],ignore_index=True)
#         print(df1.sample(df3.shape[0],replace=False).shape,df3.shape)
        randCorr=df1[featColNames].sample(df3.shape[0],replace=False).reset_index(drop=True).\
    corrwith(df3[featColNames], axis = 1,method='pearson',drop=True).values.tolist()

#         x1=df1.sample(df3.shape[0],replace=False).values
    
#         randCorr=pearsonr()
#         randCorr = [x for x in randCorr if str(x) != 'nan']
        randC=randC+randCorr
#     print(randC)    
    randC_v2=[]    
    for i in range(10):
        uniqeSamplesFromEachPurt=inDf.groupby(pertColName)[featColNames].apply(lambda s: s.sample(1))
        corrMatAcrossPurtbs=uniqeSamplesFromEachPurt.loc[:,featColNames].T.corr()
        randCorrVals=list(corrMatAcrossPurtbs.values[np.triu_indices(corrMatAcrossPurtbs.shape[0], k = 1)])
    randC_v2=randC_v2+randCorrVals
    
        
    if plotEnabled:
        fig, axes = plt.subplots(figsize=(5,3))
        sns.kdeplot(randC, bw=.1, label="random pairs",ax=axes)
        sns.kdeplot(repC, bw=.1, label="replicate pairs",ax=axes);axes.set_xlabel('CC');
        sns.kdeplot(randC_v2, bw=.1, label="random v2 pairs",ax=axes);axes.set_xlabel('CC');
#         perc5=np.percentile(repCC, 50);axes.axvline(x=perc5,linestyle=':',color='darkorange');
#         perc95=np.percentile(randCC, 90);axes.axvline(x=perc95,linestyle=':');
        axes.legend();#axes.set_title('');
        axes.set_xlim(-1.1,1.1)
    return [randC,repC,randC_v2]



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
            repCorr=np.sort(np.unique(df1.loc[:,df1.columns[df1.columns.str.contains("Metadata")!=True]].T.corr().values))[:-1].tolist()
            repC=repC+repCorr
            randAllels=df2[pertName].drop_duplicates().sample(df1.shape[0],replace=True).tolist()
            df3=pd.concat([df2[df2[pertName]==i].reset_index(drop=True).iloc[0:1,:] for i in randAllels],ignore_index=True)
            randCorr=df1.corrwith(df3, axis = 1,method='pearson').values.tolist()
            randC=randC+randCorr

        corrAll.append([randC,repC]);
    return corrAll