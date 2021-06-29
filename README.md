# Rosetta
Translating morphology data across single-cell modalities (CZI)

### In this repository:

## Preprocessed publicly available Profiles:

- We have gathered the following five available data sets that had both Cell Painting morphological (CP) and L1000 gene expression (GE) profiles, preprocessed the data from different sources and in different formats in a unified .csv format. Documentation datasets preprocessing is can be find 
    - CDRP-BBBC047-Bray-CP-GE (Cell line: U2OS)
    - CDRPBIO-BBBC036-Bray-CP-GE (Cell line: U2OS)
    - LUAD-BBBC041-Caicedo-CP-GE (Cell line: A549)
    - TA-ORF-BBBC037-Rohban-CP-GE (Cell line: U2OS)
    - LINCS-Pilot1-CP-GE (Cell line: A549)
        
- Preprocessed profiles are available on a S3 bucket (s3://cellpainting-datasets/Rosetta-GE-CP). They can be downloaded using the command:

```bash
aws s3 cp \
  --recursive \
  s3://cellpainting-datasets/Rosetta-GE-CP .  
```
    
 ## Analysis / Experiments:
 
 - We have explored predictably of GE profiles from CP morphology profiles, and vice versa
 
 - We have explored predictability of single landmark gene by CP profiles and single CP feature by GE profiles.
 
 - We have explored modality integration/augmentation methods for improving profile information content for profiling tasks such as, MOA prediction.
 
 - We have explored possibility of generating single cell images given GE treatment level profiles.
 
