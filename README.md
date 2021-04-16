Genopy

A project aiming to generate data to be used in COVID strain classification. The model is based on building a classifier using both ResNet and a Multi-layer perceptron. This is then integrated into a GAN, which is further transformed into a cGAN.

This allows avoiding the use of patient data, which can be a major setback in research efforts. Additionally, it allows for exploratory data analysis in detecting new emergent COVID strains, as the cGAN can produce similar GANs to specific continents. Further research could be the integration of an evolutionary neural network model to study COVID evolution and strain generation.

**File organization:**

covid_d3.R: Conducting preliminary data analysis

UMAP_Projection.ipynb: UMAP generation

MLP_classifier: classification system using MLP

ResNet_classifier: classification system using ResNet

D3_GAN: GAN

D3_cGAN: cGAN

