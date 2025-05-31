import polars as pl
import decoupler as dc
import numpy as np
import torch



# dc.op works for those with updated decoupler package
regulons = dc.op.collectri(organism='human')



# rows should correspond to genes
# cols should correspond to gene sets/TFs
def create_mask(pbmc_train, pbmc_val, tf_list, extra_nodes_count=1):
    '''
    Creates a binary mask DataFrame and filtered AnnData objects for selected TFs.

    Parameters
    ----------
    pbmc_train : AnnData
        Training AnnData object containing gene expression data.
    pbmc_val : AnnData
        Validation AnnData object containing gene expression data.
    tf_list : list of str
        List of transcription factors (TFs) to include in the mask.
    extra_nodes_count : int, optional (default=1)
        Number of additional unannotated nodes to add as columns to the mask.

    Returns
    -------
    mask_df : polars.DataFrame
        DataFrame of shape (n_genes, n_TFs + extra_nodes_count), where each column is a TF or unannotated node,
        and each row is a gene. Entries are 1 if the gene is regulated by the TF, 0 otherwise.
    train : AnnData
        Filtered training AnnData object containing only genes present in mask_df.
    valid : AnnData
        Filtered validation AnnData object containing only genes present in mask_df.

    Notes
    -----
    - All negative weights in the regulons are converted to 1, and positive weights are kept as is.
    - Only genes present in pbmc_train are included in the mask.
    - Extra columns named "unannotated_X" are added with all entries set to 1.
    '''
    filtered_regulons = regulons[regulons['source'].isin(tf_list)]
    
    # Ensure correct data types for compatibility
    filtered_regulons = filtered_regulons.astype({
        "source": str,
        "target": str,
        "weight": np.float64
    })

    tmp = (
        pl
        .from_pandas(filtered_regulons)
        # Convert all -1 weights to 1, remove positive weights restriction
        .with_columns(
            pl.col("weight").replace(-1, 1)
        )
        .filter(
            pl.col("target").is_in(pbmc_train.var.index.to_numpy())
        )
        .pivot(
            on="source",  # new columns (TFs)
            index="target",  # rows (genes)
            values="weight"
        )
        .fill_null(0)
    )

    # Add extra unannotated nodes as columns
    for i in range(0, extra_nodes_count):
        node_index = i + 1
        node_name = "unannotated_" + str(node_index)
        tmp = tmp.with_columns(
            pl.lit(1).alias(node_name)
        )

    # Filter AnnData objects to include only genes present in the mask
    train = pbmc_train[:, tmp["target"].to_list()].copy()
    valid = pbmc_val[:, tmp["target"].to_list()].copy()

    return tmp, train, valid

# -----------------------------------------------------------------------------
# Mask Creation Example
# -----------------------------------------------------------------------------
# Example usage: creates mask and filtered AnnData objects for selected TFs.
#tf_list
#mask, train, valid = create_mask(pbmc_train, pbmc_val, tf_list)