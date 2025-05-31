import torch
from device import device

def trainVEGA_bayes(vae, data, val_data, epochs=100, beta_en = 0.0001, beta_de = 0.0001, learning_rate = 0.001):
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
            kl_decoder = vae.decoder.sparse_layer.kl_divergence()

            loss = ((x - x_hat)**2).sum() + beta_en* kl_encoder + beta_de*kl_decoder
            loss.backward()
            opt.step()
            train_loss_e += loss.to('cpu').detach().numpy()
            #vae.decoder.weights()
            vae.encoder.clamp_mu()
        train_losses.append(train_loss_e/(len(data)*128))

        #### Here you should add the validation loop
        vae.eval()

        with torch.no_grad():
            for x in val_data:
                x = x.to(device)
                x_hat = vae(x)
                kl_encoder = vae.encoder.kl
                kl_decoder = vae.decoder.sparse_layer.kl_divergence()
                loss = ((x - x_hat)**2).sum() + beta_en* kl_encoder + beta_de*kl_decoder
                valid_loss_e += loss.to('cpu').detach().numpy()
            valid_losses.append(valid_loss_e/(len(val_data)*128))

            if epoch % 10 == 0:
                print("epoch: ", epoch, " train_loss: ", train_loss_e/(len(data)*128), "  valid_loss: ", valid_loss_e/(len(val_data)*128))

    return vae, train_losses, valid_losses