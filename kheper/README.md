# Kheper - Transforming Cell Painting and Gene Expression

As part of the Rosetta project, Kheper represents an approach to transform [cell painting](https://doi.org/10.1038/nprot.2016.105) profiles (morphology) into [L1000 profiles](https://doi.org/10.1016/j.cell.2017.10.049) (gene expression) and vice versa.
The approach is based on the [UNIT framework](https://github.com/mingyuliutw/UNIT) ([Liu et al. 2018)](https://arxiv.org/abs/1703.00848) which assumes that two distinct domains can be mapped to a shared latent space.
The two domains do not need matched profiles.

Named after the hieroglyph representing transformation, Kheper is used to study the connections between morphology and gene expression and to predict profiles across domains.

## Shared Latent Space Model

The model consists of two variational autoencoders (VAE) ([Kingma and Welling 2014](https://arxiv.org/abs/1312.6114)) mapping cell painting and L1000 gene expression profiles into a shared latent space.
Therefore, the model is trained by minimizing the reconstruction and Kullback-Leibler (KL) divergence loss between both VAE's jointly.
In addition, the model minimizes two adversarial loss functions with generative adversarial network (GAN) ([Goodfellow et al. 2014](https://arxiv.org/abs/1406.2661)) style discriminators used to distinguish (1) real morphological profiles from transformed morphological profiles and (2) real gene expression profiles from transformed gene expression profiles.
Lastly, the model minimizes two cycle loss functions, which imposes VAE losses on domains transformed into alternative domains and back again.

## Reproduce Analysis

To reproduce our analysis, first build the included conda environment.

```bash
# conda version > 4.6
conda env create --force --file environment.yml
```

Our analysis consists of several modules, which are expected to be executed sequentially.
The full analysis can be reproduced by running:

```bash
# Ensure kheper environment is activated
conda activate kheper

# Run entire pipeline
bash analysis_pipeline.sh
```

The modules include:

| Module | Description |
| :----- | :---------- |
| 0.process-data | Downloads and processes L1000 gene expression and cell painting data |

