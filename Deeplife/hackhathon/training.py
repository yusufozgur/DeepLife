import torch 
import numpy as np
import pandas as pd
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
import polars as pl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_losses(train_losses, valid_losses, path):
    with open(f"{path}_losses.csv", "w") as f:
        f.write("epoch,train_loss,valid_loss\n")
        for epoch, (train_loss, valid_loss) in enumerate(zip(train_losses, valid_losses)):
            f.write(f"{epoch},{train_loss},{valid_loss}\n")

def run_vega_model(VEGA, model_type, train_data, valid_data, mask_df, path_to_save, cond = 'all', N=10, epochs = 60, encoder = None): # I need this encoder argument 
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

    # set up dataloader
    trainX = torch.utils.data.DataLoader(train_data.X.toarray(), batch_size=128, shuffle=True)
    validX = torch.utils.data.DataLoader(valid_data.X.toarray(), batch_size=128, shuffle=True)

    all_weights = []

    match model_type:
        case "vega":
            for _ in range(N):
                vega = VEGA(mask.T, encoder = encoder, bayesian = False, dropout = 0.3, z_dropout = 0.3).to(device)     #   REPLACE THIS WITH YOUR VEGA MODEL
                vega, train_losses, valid_losses = trainVEGA_with_valid(vega, trainX, validX, epochs=epochs, beta=0.0001)
                weight = get_weight(vega)
                all_weights.append(weight)
                save_losses(train_losses, valid_losses, path_to_save) # there is no need to plot it, let's save a loss function to plot it later

        case "bayes":
            weight_uncertainties = []
            for _ in range(N):
                vega = VEGA(mask.T, encoder = encoder, bayesian = True, dropout = 0.3, z_dropout = 0.3).to(device)              #   REPLACE THIS WITH YOUR VEGA MODEL
                vega, train_losses, valid_losses = trainVEGA_with_bayes(vega, trainX, validX, epochs = epochs, beta_en = 0.0001, beta_de=0.0001)
                weight = get_weight_bayes(vega)
                uncertainty = get_weight_uncertainties_bayes(vega)
                all_weights.append(weight)
                weight_uncertainties.append(uncertainty)
                save_losses(train_losses, valid_losses, path_to_save)

            weight_uncertainties = np.stack(weight_uncertainties)
            mean_weight_uncertainties = weight_uncertainties.mean(axis=0)
            mean_weight_uncertainties = pd.DataFrame(mean_weight_uncertainties, index = train_data.var.index.tolist(), columns=mask_df.columns[1:])
            mean_weight_uncertainties.to_csv(f"{cond}_{path_to_save}_{model_type}_weight_uncertainty_mean.csv", index=True)

        case "swa":
            for _ in range(N):
                vega = VEGA(mask.T, encoder = encoder, bayesian = False, dropout = 0.3, z_dropout = 0.3).to(device)            #   REPLACE THIS WITH YOUR VEGA MODEL

                swa_model, train_losses, valid_losses = trainVEGA_with_swa(
                    vega, trainX, validX,
                    epochs=epochs, beta=0.0001,
                    learning_rate=0.0005,
                    swa_start=80,
                    swa_lr=0.0001        
                )

                W_swa = get_weight_swa(swa_model)
                all_weights.append(W_swa)
                save_losses(train_losses, valid_losses, path_to_save)


    stacked = np.stack(all_weights)
    mean_weight = stacked.mean(axis=0)
    std_weight = stacked.std(axis=0)

    mean_weight = pd.DataFrame(mean_weight, index = train_data.var.index.tolist(), columns=mask_df.columns[1:])
    std_weight = pd.DataFrame(std_weight, index = train_data.var.index.tolist(), columns=mask_df.columns[1:])


    mean_weight.to_csv(f"{cond}_{path_to_save}_{model_type}_weight_mean.csv", index=True)
    std_weight.to_csv(f"{cond}_{path_to_save}_{model_type}_weight_std.csv", index=True)

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

def trainVEGA_with_swa(vae, data, val_data, epochs=100, beta=0.0001,
                       learning_rate=0.0005, swa_start=75, swa_lr=0.05):
    opt = torch.optim.Adam(vae.parameters(), lr=learning_rate, weight_decay=5e-4)
    swa_model = AveragedModel(vae)           # will hold the running average
    swa_scheduler = SWALR(opt, swa_lr=swa_lr)

    train_losses, valid_losses = [], []
    for epoch in range(epochs):
        vae.train()
        train_loss_e = 0.
        for x in data:
            x = x.to(device)
            opt.zero_grad()
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + beta * vae.encoder.kl
            loss.backward()
            opt.step()
            vae.decoder.positive_weights()
            train_loss_e += loss.detach().cpu().item()

        # start updating our SWA weights
        if epoch >= swa_start:
            swa_model.update_parameters(vae)
            swa_scheduler.step()

        train_losses.append(train_loss_e / (len(data) * data.batch_size))

        # validation
        vae.eval()
        valid_loss_e = 0.
        with torch.no_grad():
            for x in val_data:
                x = x.to(device)
                x_hat = vae(x)
                loss = ((x - x_hat)**2).sum() + beta * vae.encoder.kl
                valid_loss_e += loss.detach().cpu().item()
        valid_losses.append(valid_loss_e / (len(val_data) * val_data.batch_size))

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_losses[-1]:.4f}, valid_loss={valid_losses[-1]:.4f}")

    # recompute batch‐norm statistics for the SWA model (if you had any)
    update_bn(data, swa_model, device=device)

    return swa_model, train_losses, valid_losses

def trainVEGA_with_valid(vae, data, val_data, epochs=100, beta = 0.0001, learning_rate = 0.001):
    opt = torch.optim.Adam(vae.parameters(), lr = learning_rate, weight_decay = 5e-4)
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        train_loss_e = 0
        valid_loss_e = 0
        vae.train() #train mode

        for x in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + beta* vae.encoder.kl
            loss.backward()
            opt.step()
            train_loss_e += loss.to('cpu').detach().numpy()
            vae.decoder.positive_weights() # we restrict the decoder to positive weights
        train_losses.append(train_loss_e/(len(data)*128))

        #### Here you should add the validation loop
        vae.eval()

        for x in val_data:
            x = x.to(device)
            x_hat = vae(x)
            loss = ((x - x_hat)**2).sum() + beta * vae.encoder.kl
            valid_loss_e += loss.to('cpu').detach().numpy()
        valid_losses.append(valid_loss_e/(len(val_data)*128))

        if epoch % 10 == 0:
            print("epoch: ", epoch, " train_loss: ", train_loss_e/(len(data)*128), "  valid_loss: ", valid_loss_e/(len(val_data)*128))

    return vae, train_losses, valid_losses

def get_weight_bayes(model):
    # pull out the sparse‐layer weight matrix
    W = model.decoder.sparse_layer[0].weight_mu.data.cpu().numpy()
    return W

def get_weight_uncertainties_bayes(model):
    # pull out the sparse‐layer weight matrix
    W = model.decoder.sparse_layer[0].weight_log_sigma.data.cpu().numpy()
    return W


def trainVEGA_with_bayes(vae, data, val_data, epochs=100, beta_en = 0.0001, beta_de = 0.0001, learning_rate = 0.001):
    opt = torch.optim.Adam(vae.parameters(), lr = learning_rate, weight_decay = 5e-4)
    train_losses = []
    valid_losses = []
    for epoch in range(epochs):
        train_loss_e = 0
        valid_loss_e = 0
        vae.train() #train mode

        for x in data:
            x = x.to(device) # GPU
            opt.zero_grad()
            x_hat = vae(x)
            kl_encoder = vae.encoder.kl
            kl_decoder = vae.decoder.kl_divergence()

            loss = ((x - x_hat)**2).sum() + beta_en* kl_encoder + beta_de*kl_decoder
            loss.backward()
            opt.step()
            train_loss_e += loss.to('cpu').detach().numpy()
            vae.decoder.positive_weights() # we restrict the decoder to positive weights
        train_losses.append(train_loss_e/(len(data)*128))

        #### Here you should add the validation loop
        vae.eval()

        for x in val_data:
            x = x.to(device)
            x_hat = vae(x)
            kl_encoder = vae.encoder.kl
            kl_decoder = vae.decoder.kl_divergence()
            loss = ((x - x_hat)**2).sum() + beta_en* kl_encoder + beta_de*kl_decoder
            valid_loss_e += loss.to('cpu').detach().numpy()
        valid_losses.append(valid_loss_e/(len(val_data)*128))

        if epoch % 10 == 0:
            print("epoch: ", epoch, " train_loss: ", train_loss_e/(len(data)*128), "  valid_loss: ", valid_loss_e/(len(val_data)*128))

    return vae, train_losses, valid_losses