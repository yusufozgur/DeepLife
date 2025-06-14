import torch 
import numpy as np
import pandas as pd
import polars as pl
from training.train_default import trainVEGA_default
from training.train_swa import trainVEGA_swa
from training.train_bayes import trainVEGA_bayes
import os
from device import device
from model.vega import VEGA
from model.decoder_default import DecoderVEGA
from model.encoder_with_mu_clamp import Encoder

def save_losses(train_losses, valid_losses, path):
    with open(os.path.join(path, "losses.csv"),"w") as f:
        f.write("epoch,train_loss,valid_loss\n")
        for epoch, (train_loss, valid_loss) in enumerate(zip(train_losses, valid_losses)):
            f.write(f"{epoch},{train_loss},{valid_loss}\n")

def one_iter_weights_or_uncertainty_to_df(mask_df, weight_or_uncertainty, iter_count):
    tfs = mask_df.drop("target").columns
    weights_per_iteration = pl.from_numpy(weight_or_uncertainty)
    weights_per_iteration.columns = tfs
    weights_per_iteration = weights_per_iteration.with_columns(
        mask_df["target"]
    )

    weights_per_iteration = weights_per_iteration.unpivot(
        index = "target", # stays in rows
        variable_name="tf",
        value_name="weight"
    )

    weights_per_iteration = weights_per_iteration.with_columns(
        pl.lit(iter_count).alias("iteration")
        )
    
    return weights_per_iteration

def run_vega_model(model_type, train_data, valid_data, mask_df, path_to_save = "save_model/vega", cond = 'all', cell_type = 'all', N=10, epochs = 60):
    """
    cond is "stimulated" or "control" or 'all' (I need 'all' to train the encoder before frezeing it)
    model type is 'bayes', 'vega' or 'swa'
    """

    # convert mask polars to DF
    numeric_columns = [
    name for name, dtype in mask_df.schema.items()
    if dtype in (pl.Float32, pl.Float64, pl.Int32, pl.Int64)
    ]
    mask = mask_df.select(numeric_columns).to_numpy()
    mask = torch.from_numpy(mask).float()

    # create mask
    if not cond == 'all':
        train_data = train_data[train_data.obs["condition"] == cond]
        valid_data = valid_data[valid_data.obs["condition"] == cond]

    if not cell_type == 'all':
        train_data = train_data[train_data.obs["cell_type"] == cell_type]
        valid_data = valid_data[valid_data.obs["cell_type"] == cell_type]

    # set up dataloader
    trainX = torch.utils.data.DataLoader(train_data.X.toarray(), batch_size=128, shuffle=True)
    validX = torch.utils.data.DataLoader(valid_data.X.toarray(), batch_size=128, shuffle=True)

    all_weights = []

    match model_type:
        case "vega":
            weights_per_iteration_set = []
            for iter_count in range(N):
                os.makedirs(path_to_save, exist_ok=True)  

                mask_np = mask_df.drop("target").to_numpy()

                latent_dims= mask_np.shape[1]
                input_dims = mask_np.shape[0]
                dropout = 0.3
                z_dropout = 0.5

                vega = VEGA(
                    encoder=Encoder(latent_dims, input_dims, dropout, z_dropout),
                    decoder=DecoderVEGA(mask_np.T)
                ).to(device)

                vega, train_losses, valid_losses = trainVEGA_default(vega, trainX, validX, epochs=epochs, beta=0.0001)
                weight = get_weight(vega)
                
                #save for kerem, so they can do stat tests
                
                weights_per_iteration_set.append(
                    one_iter_weights_or_uncertainty_to_df(
                        mask_df,
                        weight,
                        iter_count
                    )
                )

                #
                all_weights.append(weight)
                save_losses(train_losses, valid_losses, path_to_save) # there is no need to plot it, let's save a loss function to plot it later


        case "bayes":
            weights_per_iteration_set = []
            weight_uncertainties_per_iteration_set = []

            weight_uncertainties = []
            for iter_count in range(N):
                from model.decoder_bayes import DecoderBayes

                mask_np = mask_df.drop("target").to_numpy()

                latent_dims= mask_np.shape[1]
                input_dims = mask_np.shape[0]
                dropout = 0.3
                z_dropout = 0.5

                vega_bayes = VEGA(
                    encoder=Encoder(latent_dims, input_dims, dropout, z_dropout),
                    decoder=DecoderBayes(mask_np.T)
                ).to(device)
                vega, train_losses, valid_losses = trainVEGA_bayes(vega_bayes, trainX, validX, epochs = epochs, beta_en = 0.0001, beta_de=0.0001)
                weight,uncertainty  = get_weight_bayes(vega)
               
                # save for kerem, so they can do stat tests

                weights_per_iteration_set.append(
                    one_iter_weights_or_uncertainty_to_df(
                        mask_df,
                        weight,
                        iter_count
                    )
                )
                
                weight_uncertainties_per_iteration_set.append(
                    one_iter_weights_or_uncertainty_to_df(
                        mask_df,
                        uncertainty,
                        iter_count
                    )
                )
                #


                all_weights.append(weight)
                weight_uncertainties.append(uncertainty)
                save_losses(train_losses, valid_losses, path_to_save)

            weight_uncertainties = np.stack(weight_uncertainties)
            mean_weight_uncertainties = weight_uncertainties.mean(axis=0)
            mean_weight_uncertainties = pd.DataFrame(mean_weight_uncertainties, index = train_data.var.index.tolist(), columns=mask_df.columns[1:])
            mean_weight_uncertainties.to_csv(os.path.join(path_to_save, f"{cond}_{model_type}_{cell_type}_weight_uncertainty_mean.csv"), index=True)

        case "swa":
            for _ in range(N):
                os.makedirs(path_to_save, exist_ok=True)  
                swa_model, train_losses, valid_losses = trainVEGA_swa(
                    vega, trainX, validX,
                    epochs=epochs, beta=0.0001,
                    learning_rate=0.0005,
                    swa_start=80,
                    swa_lr=0.0001        
                )

                W_swa = get_weight_swa(swa_model)
                all_weights.append(W_swa)
                save_losses(train_losses, valid_losses, path_to_save)


    #save the iteration weights
    weights_per_iteration_single_df = pl.concat(weights_per_iteration_set)
    weights_per_iteration_single_df.write_csv(
        os.path.join(path_to_save, f"{cond}_{model_type}_{cell_type}_weights_iterations.tsv"),
            separator = "\t"
    )
    if model_type == "bayes":
        uncertainties_per_iteration_single_df = pl.concat(weight_uncertainties_per_iteration_set)
        uncertainties_per_iteration_single_df.write_csv(
            os.path.join(path_to_save, f"{cond}_{model_type}_{cell_type}_uncertainties_iterations.tsv"),
            separator = "\t"
        )
    #
    stacked = np.stack(all_weights)
    mean_weight = stacked.mean(axis=0)
    std_weight = stacked.std(axis=0)

    mean_weight = pd.DataFrame(mean_weight, index = train_data.var.index.tolist(), columns=mask_df.columns[1:])
    std_weight = pd.DataFrame(std_weight, index = train_data.var.index.tolist(), columns=mask_df.columns[1:])


    mean_weight.to_csv(os.path.join(path_to_save, f"{cond}_{model_type}_{cell_type}_weight_mean.csv"), index=True)
    std_weight.to_csv(os.path.join(path_to_save, f"{cond}_{model_type}_{cell_type}_weight_std.csv"), index=True)

    return vega, mean_weight, std_weight


def get_weight(vae):
  vae.eval()
  return vae.decoder.sparse_layer[0].weight.data.cpu().numpy()


def get_weight_swa(swa_model):
    """
    Extract the sparse‐decoder weights from an AveragedModel.
    """
    # if it's wrapped in AveragedModel, the averaged params live in swa_model.module
    model = swa_model.module if hasattr(swa_model, "module") else swa_model
    # pull out the sparse‐layer weight matrix
    W = model.decoder.sparse_layer[0].weight.data.cpu().numpy()
    return W

def get_weight_bayes(model):
    # pull out the sparse‐layer weight matrix
    weight_mu= model.decoder.sparse_layer.weight_mu.data.clone()
    mask = weight_mu.abs() < 1e-7

    # Apply mask: zero out corresponding mus and logvars
    weight_mu[mask] = 0.0
    # pull out the sparse‐layer weight matrix
    weight_logvar= model.decoder.sparse_layer.weight_logvar.data.clone()
    weight_logvar[mask] = -float('inf')  
    weight_std = torch.exp(0.5 * weight_logvar)
    return weight_mu.data.cpu().numpy(), weight_std.data.cpu().numpy()

#   def get_weight_bayes(model):
#       # pull out the sparse‐layer weight matrix
#       W = model.decoder.sparse_layer[0].weight_mu.data.cpu().numpy()
#       return W

def get_weight_uncertainties_bayes(model):
    # pull out the sparse‐layer weight matrix
    W = model.decoder.sparse_layer[0].weight_log_sigma.data.cpu().numpy()
    return W


