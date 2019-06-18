#!/usr/bin/env python
# coding: utf-8

# # Process LINCS Cell Painting Data into Training and Testing Data
# 
# Stratify all A549 profiles we have into 85% training and and 15% testing data.
# The data are balanced across perturbations.

# In[1]:


import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


np.random.seed(123)


# In[3]:


test_proportion = 0.15


# In[4]:


project_name = "2015_10_05_DrugRepurposing_AravindSubramanian_GolubLab_Broad"
project_dir = os.path.join("~", "bucket", "projects", project_name)

batch_name = "2016_04_01_a549_48hr_batch1"
data_dir = os.path.join(project_dir, "workspace", "backend", batch_name)


# In[5]:


file = os.path.join(data_dir, "{}.csv".format(batch_name))
df = pd.read_csv(file, low_memory=False)

# For some reason, some metadata is missing
df.loc[df.Metadata_pert_id == "BRD-K60230970", "Metadata_pert_iname"] = "MG-132"
df.loc[df.Metadata_pert_id == "BRD-K50691590", "Metadata_pert_iname"] = "bortezomib"
df.loc[df.Metadata_broad_sample == "DMSO", ["Metadata_pert_iname", "Metadata_pert_id"]] = "DMSO"

print(df.shape)
df.head(2)


# In[6]:


# Load Additional Annotations
folder = "2016_04_01_a549_48hr_batch1_CellPainting_CPfeats_whitened"
meta_dir = os.path.join(project_dir, "workspace", "metadata", folder)

file = os.path.join(meta_dir, "level_4_col_meta_n52223.txt")
annot_df = pd.read_csv(file, sep='\t')


# In[7]:


annot_df.cell_id.value_counts()


# In[8]:


train_x, test_x = train_test_split(df, test_size=test_proportion, stratify=df.Metadata_pert_id)


# ## Output Training and Testing Data

# In[9]:


print(train_x.shape)

file = os.path.join("data", "cp_train_data.tsv.gz")
train_x.to_csv(file, sep='\t', index=False)


# In[10]:


print(test_x.shape)

file = os.path.join("data", "cp_test_data.tsv.gz")
test_x.to_csv(file, sep='\t', index=False)

