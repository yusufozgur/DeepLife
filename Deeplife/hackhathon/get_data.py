# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------
# Loads PBMC training and validation datasets from .h5ad files.


import scanpy as sc

# If you do not have the files, uncomment the wget commands to download them.
#!wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zHJKoU8QcQB4cLR-oICO2YY4Nu-QaZHG' -O PBMC_train.h5ad 
pbmc_train = sc.read_h5ad("PBMC_train.h5ad")
#!wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rJKZYIG7rv7BQbDD9RElYOgHcxxzjcYj' -O PBMC_valid.h5ad # in case you don't have validation dataset downloaded 
pbmc_val = sc.read_h5ad("PBMC_valid.h5ad") 