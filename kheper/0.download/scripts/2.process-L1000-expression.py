#!/usr/bin/env python
# coding: utf-8

# # Process LINCS Gene Expression Data into Training and Testing Data
# 
# Subset L1000 Gene Expression data to A549 and the Perturbations captured in the Cell Painting Pilot.
# Stratify these profiles we have into 85% training and and 15% testing data.
# The data are balanced across perturbations.

# In[1]:


import os
import numpy as np
import pandas as pd

from cmapPy.pandasGEXpress import parse
from sklearn.model_selection import train_test_split


# In[2]:


np.random.seed(123)


# In[3]:


test_proportion = 0.15


# In[4]:


# Read only the gene names and first plate to save memory
file = os.path.join("data", "GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx")
df = parse.parse(file, cidx=[0, 1])


# In[5]:


# Load Genes
gene_file = os.path.join("data", "GSE92742_Broad_LINCS_gene_info.txt.gz")
gene_df = pd.read_csv(gene_file, sep='\t', index_col=0)

landmark_gene_df = gene_df.query("pr_is_lm == 1")
landmark_gene_df.head(2)


# ## Subset L1000 to Landmark Genes

# In[6]:


landmark_indices = list(np.where(df.data_df.index.isin(landmark_gene_df.index.astype(str)))[0])


# In[7]:


df = parse.parse(file, ridx=landmark_indices).data_df

print(df.shape)
df.head(2)


# ## Align with Cell Painting Data

# In[8]:


project_name = "2015_10_05_DrugRepurposing_AravindSubramanian_GolubLab_Broad"
project_dir = os.path.join("~", "bucket", "projects", project_name)

batch_name = "2016_04_01_a549_48hr_batch1"
data_dir = os.path.join(project_dir, "workspace", "backend", batch_name)


# In[9]:


cp_file = os.path.join(data_dir, "{}.csv".format(batch_name))
cp_df = pd.read_csv(cp_file, low_memory=False)

cp_df.loc[cp_df.Metadata_pert_id == "BRD-K60230970", "Metadata_pert_iname"] = "MG-132"
cp_df.loc[cp_df.Metadata_pert_id == "BRD-K50691590", "Metadata_pert_iname"] = "bortezomib"
cp_df.loc[cp_df.Metadata_broad_sample == "DMSO", ["Metadata_pert_iname", "Metadata_pert_id"]] = "DMSO"

print(cp_df.shape)
cp_df.head(2)


# In[10]:


# Load Experiment data
experiment_file = os.path.join("data", "GSE92742_Broad_LINCS_inst_info.txt.gz")
experiment_df = pd.read_csv(experiment_file, sep='\t', low_memory=False)

print(experiment_df.shape)
experiment_df.head()


# ## Subset to Perturbations measured in Cell Painting and A549 Profiles

# In[11]:


experiment_sub_df = (
    experiment_df
    .query("pert_id in @cp_df.Metadata_pert_id")
    .query("cell_id == 'A549'")
    .reset_index(drop=True)
)

print(experiment_sub_df.shape)
experiment_sub_df.head(2)


# In[12]:


subset_df = (
    df
    .loc[:, experiment_sub_df.inst_id]
    .transpose()
)

print(subset_df.shape)
subset_df.head(2)


# ## Output Training and Testing Data

# In[13]:


train_x, test_x = train_test_split(subset_df, test_size=test_proportion, stratify=experiment_sub_df.pert_id)


# In[14]:


print(train_x.shape)

file = os.path.join("data", "expr_train_data.tsv.gz")
train_x.to_csv(file, sep='\t', index=False)


# In[15]:


print(test_x.shape)

file = os.path.join("data", "expr_test_data.tsv.gz")
test_x.to_csv(file, sep='\t', index=False)

