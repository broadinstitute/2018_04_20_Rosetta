#!/usr/bin/env python
# coding: utf-8

# # Download and Process L1000 Gene Expresison Data
# 
# Here, I download the official L1000 Gene Expression data (**PHASE I**) from [GSE92743](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742).
# I download level 4 data subset to landmark genes.
# 
# More information about the data can be viewed [here](https://clue.io/connectopedia/guide_to_geo_l1000_data)

# In[1]:


import os
import sys
import pandas as pd

sys.path.append("..")
from scripts.download_utils import download_data


# In[2]:


base_url = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE92nnn/GSE92742/suppl/"
base_dir = "data"


# ## Dowload Gene Expression Data
# 
# Level 4 (normalized data)

# In[3]:


file_name = "GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx.gz"

download_data(base_url=base_url,
              name=file_name,
              folder=base_dir,
              append_url=True)


# ### Uncompress File

# In[4]:


get_ipython().system(' gunzip "data/GSE92742_Broad_LINCS_Level4_ZSPCINF_mlr12k_n1319138x12328.gctx.gz"')


# ## Download Other L1000 Meta Data

# In[5]:


file_names = [
    "GSE92742_Broad_LINCS_cell_info.txt.gz",
    "GSE92742_Broad_LINCS_gene_info.txt.gz",
    "GSE92742_Broad_LINCS_pert_info.txt.gz",
    "GSE92742_Broad_LINCS_pert_metrics.txt.gz",
    "GSE92742_Broad_LINCS_inst_info.txt.gz",
]


# In[6]:


for file_name in file_names:
    # Download additional metadata
    download_data(
        base_url=base_url,
        name=file_name,
        folder=base_dir,
        append_url=True
    )

