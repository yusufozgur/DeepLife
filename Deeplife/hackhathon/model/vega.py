import torch.nn as nn

class VEGA(nn.Module):
    def __init__(self, encoder, decoder):
        super(VEGA, self).__init__()

        self.decoder = decoder
        self.encoder = encoder
        #self.encoder = Encoder(latent_dims, input_dims, dropout, z_dropout) # we use the same encoder as before (two-layer, fully connected, non-linear)
        #self.decoder = DecoderVEGA(mask)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
