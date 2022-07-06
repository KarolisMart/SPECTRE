import torch.nn as nn

class NoiseMLP(nn.Module):
    def __init__(self, input_noise_latent_dim: int, output_noise_latent_dim: int, alpha=0.2, gelu=False):
        super().__init__()
        self.input_noise_latent_dim = input_noise_latent_dim
        self.noise_latent_dim = output_noise_latent_dim

        if gelu:
            activation = nn.GELU()
        else:
            activation = nn.LeakyReLU(negative_slope=alpha)

        self.mlp_noise = nn.Sequential(
                            nn.Linear(self.input_noise_latent_dim, self.noise_latent_dim),
                            activation,
                            nn.LayerNorm(self.noise_latent_dim),
                            nn.Linear(self.noise_latent_dim, self.noise_latent_dim),
                            activation,
                            nn.LayerNorm(self.noise_latent_dim),
                            nn.Linear(self.noise_latent_dim, self.noise_latent_dim),
                            activation,
                            nn.LayerNorm(self.noise_latent_dim),
                            nn.Linear(self.noise_latent_dim, self.noise_latent_dim),
                            activation,
                            nn.LayerNorm(self.noise_latent_dim),
                        )

        for m in self.mlp_noise:
            if isinstance(m, nn.Linear):
                if gelu:
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                else:
                    nn.init.kaiming_uniform_(m.weight.data, a=alpha)
                nn.init.zeros_(m.bias.data)

    def forward(self, x):
        return self.mlp_noise(x)
