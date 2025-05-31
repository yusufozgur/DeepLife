import polars as pl
import decoupler as dc
import scanpy as sc
import numpy as np
import torch

#!wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1zHJKoU8QcQB4cLR-oICO2YY4Nu-QaZHG' -O PBMC_train.h5ad 
pbmc_train = sc.read_h5ad("PBMC_train.h5ad")
#!wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1rJKZYIG7rv7BQbDD9RElYOgHcxxzjcYj' -O PBMC_valid.h5ad # in case you don't have validation dataset downloaded 
pbmc_val = sc.read_h5ad("PBMC_valid.h5ad") 

# dc.op does not work for me
regulons = dc.op.collectri(organism='human')

filtered_regulons = regulons[regulons['source'].isin(
    [
      "GTF2I",
      "GTF3A"
      "NRF1",
      "ELF1",
      "STAT1",
      "STAT2",
      "IRF9",
      "STAT3",
      "STAT4",
      "STAT5A",
      "STAT5B",
      "IRF3",
      "IRF7",
      "IRF1",
      "IRF5",
     "IRF8",
     ])]

# rows should correspond to genes
# cols should correspond to gene sets/TFs
def create_mask(pbmc_train, pbmc_val, filtered_regulons, extra_nodes_count=1):
    '''
    filtered_regulons: collectri regulons filtered to contain only selected TFs
    pbmc_train: AnnData object
    pbmc_val: AnnData object
    extra_nodes_count: number of additional unannotated nodes to add to the mask

    Outputs:
    mask: torch tensor of shape (n_genes, n_TFs + extra_nodes_count)
    mask_df: polars DataFrame with the same shape as mask, containing the TFs as columns and genes as rows
    train: AnnData object with the same genes as mask_df
    valid: AnnData object with the same genes as mask_df
    '''
####FOR DTYPE COMPATIBILITY
    filtered_regulons = filtered_regulons.astype({
        "source": str,
        "target": str,
        "weight": np.float64
    })

    tmp = (
        pl
        .from_pandas(filtered_regulons)
        # alice and kerem and yusuf decided to convert all -1 to 1 and remove positive weights restriction
        .with_columns(
            pl.col("weight").replace(-1, 1)
        )
        .filter(
            pl.col("target").is_in(pbmc_train.var.index.to_numpy())
        )  # 901 x 13 -> 332 x 13
        .pivot(
            on="source",  # new columns
            index="target",  # stays in rows
            values="weight"
        )
        .fill_null(0)
    )

    for i in range(0, extra_nodes_count):
        node_index = i + 1
        node_name = "unannotated_" + str(node_index)
        tmp = tmp.with_columns(
            pl.lit(1).alias(node_name)
        )


    train = pbmc_train[:, tmp["target"].to_list()].copy()
    valid = pbmc_val[:, tmp["target"].to_list()].copy()

    return tmp, train, valid

mask, train, valid = create_mask(pbmc_train, pbmc_val, filtered_regulons) 