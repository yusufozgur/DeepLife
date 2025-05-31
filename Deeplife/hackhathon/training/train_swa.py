import torch
from device import device
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn

def trainVEGA_swa(vae, data, val_data, epochs=100, beta=0.0001,
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
            vae.decoder.weights()
            vae.encoder.clamp_mu()
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
