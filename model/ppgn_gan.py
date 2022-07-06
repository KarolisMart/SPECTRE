import torch
from torch import nn

from model.ppgn import Powerful
from util.model_helper import zero_diag, discretize

class PPGNGenerator(nn.Module):
    def __init__(
        self,
        alpha=0.2,
        n_max=36,
        noise_latent_dim=100,
        n_layers=8,
        data_channels=64,
        gelu=False,
        k_eigval=18,
        use_fixed_emb=False,
        normalization='instance',
        dropout=0.0,
        skip_connection=False,
        cat_eigvals=False,
        cat_mult_eigvals=False,
        no_extra_n=False,
        no_cond=False,
        init_emb_channels=64,
        qm9=False,
        data_channels_mult=1,
    ):
        super(PPGNGenerator, self).__init__()

        data_channels = data_channels * data_channels_mult

        self.n_max = n_max
        self.latent_dim = noise_latent_dim
        self.alpha = alpha
        self.n_layers = n_layers
        self.k_eigval = k_eigval
        self.use_fixed_emb = use_fixed_emb
        self.skip_connection = skip_connection
        self.cat_eigvals = cat_eigvals
        self.cat_mult_eigvals = cat_mult_eigvals
        self.no_extra_n = no_extra_n
        self.no_cond = no_cond
        self.qm9 = qm9

        self.input_features = 1 + self.latent_dim  # adjacency + noise
        if not self.no_cond and not self.use_fixed_emb:
            self.input_features += k_eigval #  + eigvecs
        if self.use_fixed_emb:
            self.input_features += init_emb_channels
        if self.cat_eigvals or self.cat_mult_eigvals:
            self.input_features += k_eigval 
        if not self.no_extra_n:
            self.input_features += 1 # + n
        
        if gelu:
            activation = nn.GELU()
        else:
            activation = nn.LeakyReLU(negative_slope=alpha)
        if self.use_fixed_emb:
            self.embedding = nn.Embedding(self.n_max, init_emb_channels)

        spectral_norm = lambda x: x

        if self.qm9:
            output_features = 4
        else:
            output_features = 1

        self.powerful = Powerful(num_layers=n_layers, input_features=self.input_features, hidden=data_channels, hidden_final=data_channels,
                                dropout_prob=dropout, simplified=False, n_nodes=self.n_max, normalization=normalization, adj_out=True,
                                output_features=output_features, residual=False, activation=activation, spectral_norm=spectral_norm,
                                node_out=self.qm9, node_output_features=4)

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if gelu:
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                else:
                    nn.init.kaiming_uniform_(m.weight.data, a=self.alpha)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, node_noise, eigval, eigvec, mask):
        n = torch.sum(mask, dim=-1, keepdim=True)
        n = n/self.n_max

        if not self.no_extra_n:
            x = [node_noise, n]
        else:
            x = [node_noise]
        
        if not self.no_cond:
            if self.use_fixed_emb:
                node_embeddings = self.embedding(torch.arange(mask.size(1), device=n.device, dtype=torch.long)).unsqueeze(0).expand(mask.size(0), -1, -1).clone()
            else:
                node_embeddings = eigvec[:,:,:self.k_eigval]
            if self.cat_eigvals or self.cat_mult_eigvals:
                x.append(eigval[:,:self.k_eigval].unsqueeze(1).expand_as(node_embeddings))
            if not self.cat_eigvals and not self.use_fixed_emb:
                eigval = eigval.clone()
                eigval[eigval==0] = eigval[eigval==0] + 1e-8 # Avoid sqrt gradient problems
                node_embeddings = node_embeddings * torch.sqrt(eigval[:,:self.k_eigval].abs()).unsqueeze(1).expand_as(node_embeddings)
            x.append(node_embeddings)

        x = torch.cat(x, dim=-1)
        del node_noise
        
        if self.use_fixed_emb or self.no_cond:
            adj = torch.zeros_like(mask)
        elif self.cat_eigvals:
            adj = (node_embeddings * eigval[:,:self.k_eigval].unsqueeze(1).expand_as(node_embeddings)) @ node_embeddings.transpose(-2,-1)
        else:
            adj = node_embeddings @ node_embeddings.transpose(-2,-1)

        if not self.no_cond:
            del node_embeddings

        if self.qm9:
            adj, node_features = self.powerful(adj, x, mask)

            node_features = node_features * mask[:,:,0].unsqueeze(-1)
            adj = (adj + adj.transpose(1, 2)) / 2
            adj = adj.softmax(-1)

            edge_features = adj[:,:,:,1:]

            adj = 1 - adj[:,:,:,0]
            adj = zero_diag(adj)
            adj = adj * mask

            if adj.isnan().any():
                print('adj', adj.isnan().any())

            edge_features = edge_features * (1 - torch.eye(edge_features.size(1), edge_features.size(2), device=edge_features.device).view(1, edge_features.size(1), edge_features.size(2), 1).expand_as(edge_features))
            edge_features = edge_features * mask.unsqueeze(-1).expand_as(edge_features)

            return adj, node_features, edge_features
        else:
            adj = self.powerful(adj, x, mask)
            del x
            adj = (adj + adj.transpose(1, 2)) / 2

            adj = adj.view(mask.shape)

            adj = adj.sigmoid()
            
            adj = zero_diag(adj)
            adj = adj * mask
            if adj.isnan().any():
                print('adj', adj.isnan().any())

            return adj


class MLPGenerator(nn.Module):
    def __init__(
        self,
        alpha=0.2,
        n_max=36,
        noise_latent_dim=100,
        n_layers=8,
        data_channels=64,
        gelu=False,
        k_eigval=18,
        discretize_adjacency=False,
        shared_weights=False,
        use_fixed_emb=False,
        skip_connection=False,
        cat_eigvals=False,
        cat_mult_eigvals=False,
        no_extra_n=False,
        use_eigvecs=False,
        qm9=False,
        data_channels_mult=1,
    ):
        super(MLPGenerator, self).__init__()
        
        data_channels = data_channels * data_channels_mult

        self.n_max = n_max
        self.latent_dim = noise_latent_dim
        self.alpha = alpha
        self.n_layers = n_layers
        self.k_eigval = k_eigval
        self.discretize_adjacency = discretize_adjacency
        self.shared_weights = shared_weights
        self.use_fixed_emb = use_fixed_emb
        self.skip_connection = skip_connection
        self.cat_eigvals = cat_eigvals
        self.cat_mult_eigvals = cat_mult_eigvals
        self.no_extra_n = no_extra_n
        self.use_eigvecs = use_eigvecs
        self.qm9 = qm9

        self.input_features = 0
        self.input_features += self.latent_dim  # noise
        if not self.no_extra_n:
            self.input_features += 1 # + n
        
        if gelu:
            activation = nn.GELU()
        else:
            activation = nn.LeakyReLU(negative_slope=alpha)

        # MOLGAN uses 3-layer MLP of [128, 256, 512] and a linear projection layer to X and A
        self.mlp = nn.Sequential(
                nn.Linear(self.input_features, data_channels*2),
                activation,
                nn.LayerNorm(data_channels*2),
                nn.Linear(data_channels*2, data_channels*4),
                activation,
                nn.LayerNorm(data_channels*4),
                nn.Linear(data_channels*4, data_channels*8),
                activation,
                nn.LayerNorm(data_channels*8),
            )
        if self.qm9:
            output_features = 4
            self.lin_X = nn.Linear(data_channels*8, self.n_max*4)
        else:
            output_features = 1
        self.lin_A = nn.Linear(data_channels*8, self.n_max*self.n_max*output_features)            

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if gelu:
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                else:
                    nn.init.kaiming_uniform_(m.weight.data, a=self.alpha)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, node_noise, eigval, eigvec, mask):
        n = torch.sum(mask, dim=-1, keepdim=True)
        n = n/self.n_max

        if not self.no_extra_n:
            x = [node_noise, n[:,0]]
        else:
            x = [node_noise]

        x = torch.cat(x, dim=-1)
        del node_noise
        
        x = self.mlp(x)
        if self.qm9:
            node_features = self.lin_X(x).view(mask.size(0), self.n_max, 4)
            node_features = node_features * mask[:,:,0].unsqueeze(-1)

            adj = self.lin_A(x).view(mask.size(0), self.n_max, self.n_max, -1)[:, :mask.size(1), :mask.size(2)]
            del x

            adj = (adj + adj.transpose(1, 2)) / 2
            adj = adj.softmax(-1)

            edge_features = adj[:,:,:,1:]

            adj = 1 - adj[:,:,:,0]
            adj = zero_diag(adj)
            adj = adj * mask

            if adj.isnan().any():
                print('adj', adj.isnan().any())

            edge_features = edge_features * (1 - torch.eye(edge_features.size(1), edge_features.size(2), device=edge_features.device).view(1, edge_features.size(1), edge_features.size(2), 1).expand_as(edge_features))
            edge_features = edge_features * mask.unsqueeze(-1).expand_as(edge_features)
            return adj, node_features, edge_features
        else:
            adj = self.lin_A(x).view(mask.size(0), self.n_max, self.n_max)[:, :mask.size(1), :mask.size(2)]
            del x
            adj = (adj + adj.transpose(1, 2)) / 2

            adj = adj.sigmoid()

            adj = zero_diag(adj)
            adj = adj * mask
            if adj.isnan().any():
                print('adj', adj.isnan().any())

            return adj


class PPGNDiscriminator(nn.Module):
    def __init__(
        self,
        alpha=0.2,
        n_max=36,
        n_layers=8,
        data_channels=64,
        use_spectral_norm=False,
        normalization='instance',
        gelu=False,
        k_eigval=18,
        dropout=0.0,
        cat_eigvals=False,
        cat_mult_eigvals=False,
        partial_laplacian=False,
        no_cond=False,
        qm9=False,
        data_channels_mult=1,
    ):
        super(PPGNDiscriminator, self).__init__()

        data_channels = data_channels * data_channels_mult

        self.batch_norm_1d = True # Always 1D
        self.alpha = alpha
        self.n_max = n_max
        self.normalization = normalization
        self.cat_eigvals = cat_eigvals
        self.cat_mult_eigvals = cat_mult_eigvals
        self.partial_laplacian = partial_laplacian
        self.no_cond = no_cond
        self.qm9 = qm9

        self.k_eigval = k_eigval

        if gelu:
            activation = nn.GELU()
        else:
            activation = nn.LeakyReLU(negative_slope=alpha)
        self.activation = activation

        if use_spectral_norm:
            spectral_norm = lambda x: nn.utils.spectral_norm(x)
        else:
            spectral_norm = lambda x: x

        input_features = 2 # n + adj
        if not self.no_cond:
            input_features += self.k_eigval # + Eigvecs
        if self.cat_eigvals or self.cat_mult_eigvals:
            input_features += self.k_eigval 
        if self.partial_laplacian:
            input_features += 1
        if self.qm9:
            input_features += 7 # 4 node features and 3 edge features stacked together
        
        self.powerful = Powerful(num_layers=n_layers, input_features=input_features, hidden=data_channels, hidden_final=data_channels,
                                dropout_prob=dropout, simplified=False, n_nodes=self.n_max, normalization=self.normalization,
                                residual=False, activation=activation, spectral_norm=spectral_norm)
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if gelu:
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                else:
                    nn.init.kaiming_uniform_(m.weight.data, a=self.alpha)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, eigval, eigvec, mask, adj, node_features=None, edge_features=None):  
        n = torch.sum(mask, dim=-1, keepdim=True)
        # Normalize n by n_max for use as features
        n = n/self.n_max

        eigval = eigval[:, :self.k_eigval]
        eigvec = eigvec[:, :, :self.k_eigval]

        if self.partial_laplacian:
            lap = (eigvec * eigval.unsqueeze(1).expand_as(eigvec)) @ eigvec.transpose(-2,-1)
            adj = torch.stack([adj, lap], dim=-1)
            del lap

        x = n
        if not self.no_cond:
            if not self.cat_eigvals:
                eigval = eigval.clone()
                eigval[eigval==0] = eigval[eigval==0] + 1e-8 # Avoid sqrt gradient problems
                eigvec = eigvec * torch.sqrt(eigval.abs()).unsqueeze(1).expand_as(eigvec)

            if self.cat_eigvals or self.cat_mult_eigvals:
                x = torch.cat([x, eigval.unsqueeze(1).expand_as(eigvec), eigvec], dim=-1)
            else:
                x = torch.cat([x, eigvec], dim=-1)

        if self.qm9:
            x = torch.cat([x, node_features], dim=-1)
            adj = torch.cat([adj.unsqueeze(-1), edge_features], dim=-1)

        x = self.powerful(adj, x, mask)
        del adj

        if x.isnan().any():
            print('x_last', x)

        return x
