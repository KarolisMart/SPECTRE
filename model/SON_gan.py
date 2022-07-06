from numpy import dtype
import torch
from torch import nn
import torch.nn.functional as F
import math

# Using householder reflections is a faster way to sample from a Stiefel manifold, but matrices are not necessarily in SO(N), but simple sign flip (* -1) should solve it.
# However,  for simplicity we don't do this here.
# from torch_householder import torch_householder_orgqr
# How householder construction would look like: (also applicable for full rotation matrix construction)
# lower = torch.tril(skew, diagonal=-1)
# lower = lower + torch.eye(lower.size(1), lower.size(2), device=lower.device)
# rot = torch_householder_orgqr(lower) * -1 # Force to be SO(N)

from util.model_helper import zero_diag, masked_layer_norm2D, stiefel_metric

class SkipLinearLayer(nn.Module):
    def __init__(self, in_features, out_features, spectral_norm=(lambda x: x), skip_connection=False):
        super(SkipLinearLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.skip_connection = skip_connection
        self.layer = nn.Sequential(spectral_norm(nn.Linear(in_features, out_features)))

    def forward(self, x):
        if self.skip_connection:
            return self.layer(x) + x
        else:
            return self.layer(x)

class SkipMLPLayer(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.ReLU(), last_norm=False, spectral_norm=(lambda x: x), skip_connection=False, act_after_norm=False):
        super(SkipMLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.skip_connection = skip_connection
        self.activation = activation
        self.act_after_norm = act_after_norm
        self.layers = []
        self.norms = []
        self.acts = []
        self.do_skip = []
        for i, out_i in enumerate(out_features):
            if i == 0:
                in_i = in_features
            else:
                in_i = out_features[i-1]
            self.do_skip.append(in_i == out_i)
            self.layers.append(nn.Sequential(spectral_norm(nn.Linear(in_i, out_i))))
            if self.skip_connection:
                self.norms.append(nn.LayerNorm(in_i))
                self.acts(self.activation)
            elif (i == (len(out_features) - 1) and not last_norm):
                self.norms.append(nn.Identity())
                self.acts.append(nn.Identity())
            else:
                self.norms.append(nn.LayerNorm(out_i))
                self.acts.append(self.activation)
        self.layers = nn.ModuleList(self.layers)
        self.norms = nn.ModuleList(self.norms)
        self.acts = nn.ModuleList(self.norms)

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(*x.shape[:-1], 1, device=x.device)
        for act, layer, norm, do_skip in zip(self.acts, self.layers, self.norms, self.do_skip):
            if self.skip_connection:
                if self.act_after_norm:
                    out = act(norm(x))
                else:
                    out = norm(act(x))
                out = layer(out)
                if do_skip:
                    out = out + x
                x = out * mask
            else:
                x = layer(x)
                if self.act_after_norm:
                    out = act(norm(x))
                else:
                    out = norm(act(x))
                x = out * mask
        return x

class PointNetST(nn.Module):
    def __init__(self, in_features, out_features, data_channels, activation, last_norm=False, spectral_norm=(lambda x: x), skip_connection=False, max_pool=False):
        super(PointNetST, self).__init__()
        self.max_pool = max_pool
        self.mlp_features = SkipMLPLayer(in_features, [data_channels, data_channels, data_channels], activation=activation, last_norm=True, spectral_norm=spectral_norm, skip_connection=skip_connection, act_after_norm=False)
        self.mlp_agg = SkipMLPLayer(data_channels, [data_channels, data_channels*2, data_channels*4], activation=activation, last_norm=True, spectral_norm=spectral_norm, skip_connection=skip_connection, act_after_norm=False)
        self.mlp_cat = SkipMLPLayer(data_channels*5, [data_channels*4, data_channels*4, out_features], activation=activation, last_norm=last_norm, spectral_norm=spectral_norm, skip_connection=skip_connection, act_after_norm=False)

    def forward(self, x, mask):
        x = self.mlp_features(x, mask=mask)

        mean_x = self.mlp_agg(x, mask=mask)
        mean_x = mean_x * mask
        if self.max_pool:
            mean_x = mean_x.max(dim=1, keepdim=True)[0]
        else:
            mean_x = mean_x.sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True)
        
        x = self.mlp_cat(torch.cat([x, mean_x.expand(-1,x.size(1),-1)], dim=-1))
        del mean_x
        return x

class SONGenerator(nn.Module):
    def __init__(
        self,
        alpha=0.2,
        n_max=36,
        data_channels=128,
        gelu=False,
        k_eigval=36,
        noise_latent_dim=100,
        n_rot=1,
        mult_eigvals=False,
        dropout=0.0,
        max_pool=False,
        skip_connection=False,
        share_weights=False,
        normalize_left=False,
        no_extra_n=False,
        small=False,
        init_bank_size=10,
        gumbel_temperature=1.0,
        kl_init_scale=False,
        stiefel_sim_init=False,
    ):
        super(SONGenerator, self).__init__()

        if small:
            data_channels = data_channels // 2
        
        self.n_max = n_max
        self.data_channels = data_channels
        self.latent_dim = noise_latent_dim
        self.alpha = alpha
        self.n_rot = n_rot
        self.k_eigval = k_eigval
        self.mult_eigvals = mult_eigvals
        self.dropout = dropout
        self.max_pool = max_pool
        self.share_weights = share_weights
        self.normalize_left = normalize_left
        self.no_extra_n = no_extra_n
        self.init_bank_size = init_bank_size
        self.gumbel_temperature = gumbel_temperature
        self.kl_init_scale = kl_init_scale
        self.stiefel_sim_init = stiefel_sim_init

        if gelu:
            activation = nn.GELU()
        else:
            activation = nn.LeakyReLU(negative_slope=alpha)

        self.activation = activation

        data_channels = self.data_channels
        
        self.k_thetas = self.k_eigval*(self.k_eigval-1)//2

        if self.init_bank_size > 0:
            self.n_thetas = n_max * self.k_eigval - self.k_eigval * (self.k_eigval + 1)//2
            self.init_rot = nn.Parameter(torch.Tensor(self.init_bank_size, self.n_thetas))
            self.init_rot.requires_grad = True
            nn.init.kaiming_uniform_(self.init_rot, a=0)

            self.bank_sample_hist = torch.zeros(self.init_bank_size)
        else: # If bank size <= 0 we just use one learned initial eigenvector matrix
            self.n_thetas = n_max*(n_max-1)//2
            self.init_rot = nn.Parameter(torch.Tensor(self.n_thetas))
            self.init_rot.requires_grad = True
            nn.init.normal_(self.init_rot, mean=0.0, std=0.5)
        
        input_size = self.k_eigval

        self.node_cond_mlp = nn.Sequential(
            nn.Linear(self.k_eigval + self.latent_dim, data_channels),
            activation,
            nn.LayerNorm(data_channels),
        )

        if not self.no_extra_n:
            input_size += 1 

        if self.init_bank_size > 0:
            if self.stiefel_sim_init:
                self.init_out = n_max * self.k_eigval
            else:
                self.init_out = self.init_bank_size
            init_key_mlp_input = self.k_eigval 
            if not self.no_extra_n:
                init_key_mlp_input += 1 
            self.init_key_mlp = nn.Sequential(
                nn.Linear(init_key_mlp_input, data_channels*2),
                activation,
                nn.LayerNorm(data_channels*2),
                nn.Linear(data_channels*2, data_channels*4),
                activation,
                nn.LayerNorm(data_channels*4),
                nn.Linear(data_channels*4, self.init_out),
            )

        self.left_multiply_pointnet = []
        if self.normalize_left:
            self.left_multiply_norm_mult = []
            self.left_multiply_norm_bias = []

        self.right_multiply_pointnet = []
        self.right_multiply_mlp_theta = []

        for i in range(self.n_rot):
            if not self.share_weights or i == 0:
                # Left multiplication
                self.left_multiply_pointnet.append(PointNetST(data_channels + self.k_eigval, data_channels*4, data_channels, activation, last_norm=False, spectral_norm=(lambda x: x), skip_connection=skip_connection, max_pool=max_pool))
                # Left normalization
                if self.normalize_left:
                    left_norm_mult = nn.Parameter(torch.Tensor(1))
                    left_norm_bias = nn.Parameter(torch.Tensor(1))
                    left_norm_mult.requires_grad = True
                    left_norm_bias.requires_grad = True
                    nn.init.ones_(left_norm_mult)
                    nn.init.zeros_(left_norm_bias)
                    self.left_multiply_norm_mult.append(left_norm_mult)
                    self.left_multiply_norm_bias.append(left_norm_bias)

                # Right multiplication
                self.right_multiply_pointnet.append(PointNetST(data_channels + self.k_eigval, data_channels*4, data_channels, activation, last_norm=True, spectral_norm=(lambda x: x), skip_connection=skip_connection, max_pool=max_pool))
                self.right_multiply_mlp_theta.append(SkipMLPLayer(data_channels*4, [data_channels*2, data_channels, self.k_thetas], activation=activation, last_norm=False, spectral_norm=(lambda x: x), skip_connection=skip_connection, act_after_norm=False))

        self.left_multiply_pointnet = nn.ModuleList(self.left_multiply_pointnet)
        if self.normalize_left:
            self.left_multiply_norm_mult = nn.ParameterList(self.left_multiply_norm_mult)
            self.left_multiply_norm_bias = nn.ParameterList(self.left_multiply_norm_bias)

        self.right_multiply_pointnet = nn.ModuleList(self.right_multiply_pointnet)
        self.right_multiply_mlp_theta = nn.ModuleList(self.right_multiply_mlp_theta)

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'right_multiply_mlp_theta.' in name and '.layers.2.0' in name:
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
                elif 'left_multiply_pointnet' in name and 'mlp_cat.layers.2.0' in name and (not self.normalize_left): 
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.1)
                elif gelu:
                    nn.init.xavier_normal_(m.weight.data, gain=1.0)
                else:
                    nn.init.kaiming_normal_(m.weight.data, a=self.alpha)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias.data)

    def forward(self, node_noise, eigval, mask):

        n = torch.sum(mask, dim=-1, keepdim=True)
        n = n/self.n_max

        mask_rows = mask[:,:,1].unsqueeze(-1).float()

        if not self.no_extra_n:
            node_cond = self.node_cond_mlp(torch.cat([eigval[:,:self.k_eigval].unsqueeze(1).expand(-1, mask.size(1), -1), node_noise, n], dim=-1))
        else:
            node_cond = self.node_cond_mlp(torch.cat([eigval[:,:self.k_eigval].unsqueeze(1).expand(-1, mask.size(1), -1), node_noise], dim=-1))

        node_cond = node_cond * mask_rows

        loss = 0.0
        
        if self.init_bank_size > 0:
            lower = torch.zeros(self.init_bank_size, self.n_max, self.k_eigval, device=self.init_rot.device, dtype=self.init_rot.dtype)
            lower[torch.tril(torch.ones_like(lower), diagonal=-1) == 1] = self.init_rot.view(-1)
            lower = lower[:, :mask.size(1), :mask.size(2)]

            lower = torch.cat([lower, torch.zeros(lower.size(0), lower.size(1), lower.size(1) - lower.size(2), device=lower.device, dtype=lower.dtype)], dim=-1)
            skew = lower - lower.transpose(-2, -1)
            skew = skew.unsqueeze(0).expand(mask.size(0), -1, -1, -1).clone()
            skew = skew * mask.unsqueeze(1)
            U_bank = torch.matrix_exp(skew)
            U_bank = U_bank[:,:,:,:self.k_eigval]
            del skew
            del lower
            U_bank = U_bank * mask.unsqueeze(1)[:,:,:,:self.k_eigval]                

            if not self.no_extra_n:
                keys = torch.cat([eigval[:,:self.k_eigval], n[:,0]], dim=-1)
            else:
                keys = eigval[:,:self.k_eigval]
            if self.stiefel_sim_init:
                keys = self.init_key_mlp(keys).view(mask.size(0), 1, self.n_max, self.k_eigval)
                keys = keys[:, :mask.size(1), :mask.size(2)]
                keys = keys * mask_rows.unsqueeze(1)
                scores = stiefel_metric(keys, U_bank, manifold=U_bank)
                scores = scores * 2 / self.k_eigval # Normalize scores such that distance from U to itself is 1
                del keys
            else: # Just doing Gumbel Sampling Directly works ok, but a bit worse.
                scores = self.init_key_mlp(keys)
            one_hot = F.gumbel_softmax(scores, tau=self.gumbel_temperature, hard=True, dim=-1)
            # Track which init rots are sampled
            self.bank_sample_hist = self.bank_sample_hist.to(one_hot.device)
            self.bank_sample_hist += one_hot.sum(dim=0)
            U = torch.einsum('b d n k, b d -> b n k', U_bank, one_hot)
            if self.kl_init_scale > 0:
                qy = F.softmax(scores, dim=-1)
                loss = self.kl_init_scale * torch.sum(qy * torch.log(qy * self.init_bank_size + 1e-10), dim=-1).mean()
            del scores, one_hot
        else: # Using one fixed initial matrix in many cases is only a bit worse
            upper = torch.zeros(self.n_max, self.n_max, device=mask.device, dtype=self.init_rot.dtype)
            upper[torch.triu(torch.ones_like(upper), diagonal=1) == 1] = self.init_rot
            skew = upper - torch.transpose(upper, -2, -1)
            skew = skew.unsqueeze(0).expand(mask.size(0), -1, -1).clone()
            skew = skew[:, :mask.size(1), :mask.size(2)] * mask
            U = torch.matrix_exp(skew)
            U = U * mask
            U = U[:,:,:self.k_eigval]    

        for i in range(self.n_rot):
            if self.share_weights:
                i = 0

            # Left multiplication
            if self.mult_eigvals:
                x = U
                eigval = eigval.clone()
                eigval[eigval==0] = eigval[eigval==0]  + 1e-8 # Avoid sqrt gradient problems
                x = x * torch.sqrt(eigval[:,:self.k_eigval].abs()).unsqueeze(1).expand_as(x)
                if not self.no_extra_n:
                    x = torch.cat([x, node_cond, n], dim=-1)
                else:
                    x = torch.cat([x, node_cond], dim=-1)
            else:
                x = torch.cat([U, node_cond], dim=-1)
            
            x = self.left_multiply_pointnet[i](x, mask=mask_rows)
            x = x * mask_rows
            skew = (x @ x.transpose(-2, -1)) / math.sqrt(x.size(-1))
            if self.normalize_left:
                # Normalize the skew matrix parameters
                skew = self.left_multiply_norm_mult[i] * masked_layer_norm2D(skew.unsqueeze(-1), mask=zero_diag(mask)) + self.left_multiply_norm_bias[i] 
                skew = skew.squeeze(-1)
            skew = torch.triu(skew, diagonal=1)
            skew = skew - skew.transpose(-2, -1)
            rot = torch.matrix_exp(skew)
            U = rot @ U 
            U = U * mask_rows

            # Right multiplication
            if self.mult_eigvals:
                x = U
                eigval = eigval.clone()
                eigval[eigval==0] = eigval[eigval==0]  + 1e-8 # Avoid sqrt gradient problems
                x = x * torch.sqrt(eigval[:,:self.k_eigval].abs()).unsqueeze(1).expand_as(x)
                if not self.no_extra_n:
                    x = torch.cat([x, node_cond, n], dim=-1)
                else:
                    x = torch.cat([x, node_cond], dim=-1)
            else:
                x = torch.cat([U, node_cond], dim=-1)
            
            x = self.right_multiply_pointnet[i](x, mask=mask_rows)
            x = x * mask_rows

            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.max_pool:
                x = x.max(dim=1, keepdim=True)[0]
            else:
                x = x.sum(dim=1, keepdim=True) / mask_rows.sum(dim=1, keepdim=True)

            x = self.right_multiply_mlp_theta[i](x)

            if torch.isnan(x).any():
                print('mlp')
            
            upper = torch.zeros(mask.size(0), self.k_eigval, self.k_eigval, device=mask.device, dtype=x.dtype)
            upper[torch.triu(torch.ones_like(upper), diagonal=1) == 1] = x.reshape(-1)
            skew = upper - torch.transpose(upper, -2, -1)
            # Build rotation matrix from a skew-symetric matrix
            rot = torch.matrix_exp(skew) 
            U = U @ rot
            U = U * mask_rows

        return U, loss


class SONPointNetDiscriminator(nn.Module):
    def __init__(
        self,
        alpha=0.2,
        n_max=36,
        data_channels=128,
        use_spectral_norm=False,
        gelu=False,
        k_eigval=18,
        dropout=0.0,
        max_pool=False,
        skip_connection=False,
        full_readout=False,
        n_rot=2,
        normalize_left=False,
        small=False,
    ):
        super(SONPointNetDiscriminator, self).__init__()

        if small:
            data_channels = data_channels // 2

        self.alpha = alpha
        self.n_max = n_max
        self.dropout = dropout
        self.max_pool = max_pool
        self.full_readout = full_readout
        self.n_rot = n_rot
        self.normalize_left = normalize_left

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

        input_size = self.k_eigval + data_channels 
        self.input_mlp = nn.Sequential(
            spectral_norm(nn.Linear(self.k_eigval + 1, data_channels)),
            activation,
            nn.LayerNorm(data_channels),
        )

        self.k_thetas = self.k_eigval*(self.k_eigval-1)//2

        self.pointnet_right_1 = PointNetST(input_size, data_channels*4, data_channels, activation, last_norm=True, spectral_norm=spectral_norm, skip_connection=skip_connection, max_pool=max_pool)
        self.mlp_theta_right_1 = SkipMLPLayer(data_channels*4, [data_channels*2, data_channels, self.k_thetas], activation=activation, last_norm=False, spectral_norm=spectral_norm, skip_connection=skip_connection, act_after_norm=False)

        self.mlp_between_rights = nn.Sequential(
            spectral_norm(nn.Linear(self.k_eigval + data_channels, data_channels)),
            activation,
            nn.LayerNorm(data_channels),
            spectral_norm(nn.Linear(data_channels, data_channels)),
            activation,
            nn.LayerNorm(data_channels),
        )

        self.pointnet_right_2 = PointNetST(data_channels, data_channels*4, data_channels, activation, last_norm=True, spectral_norm=spectral_norm, skip_connection=skip_connection, max_pool=max_pool)
        self.mlp_theta_right_2 = SkipMLPLayer(data_channels*4, [data_channels*2, data_channels, (data_channels*(data_channels-1)//2)], activation=activation, last_norm=False, spectral_norm=spectral_norm, skip_connection=skip_connection, act_after_norm=False)

        self.pointnet_readout = PointNetST(data_channels, data_channels*4, data_channels, activation, last_norm=True, spectral_norm=spectral_norm, skip_connection=skip_connection, max_pool=max_pool)
        self.mlp_out = SkipMLPLayer(data_channels*4, [data_channels*2, data_channels, 1], activation=activation, last_norm=False, spectral_norm=spectral_norm, skip_connection=skip_connection, act_after_norm=False)

        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if 'mlp_theta' in name and '.layers.2.0' in name:
                    torch.nn.init.normal_(m.weight.data, mean=0.0, std=0.01)
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

    def forward(self, eigval, eigvec, mask):
        n = torch.sum(mask, dim=-1, keepdim=True)
        # Normalize n by n_max for use as features
        n = n/self.n_max

        mask_rows = mask[:,:,1].unsqueeze(-1).float()

        eigval = eigval[:, :self.k_eigval]
        eigvec = eigvec[:, :, :self.k_eigval]

        inputs = self.input_mlp(torch.cat([eigval, n[:,0]], dim=-1)).unsqueeze(1).expand(-1, eigvec.size(1), -1)

        # First right mult
        x = torch.cat([inputs, eigvec], dim=-1) 

        x = self.pointnet_right_1(x, mask=mask_rows)

        x = x * mask_rows
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.max_pool:
            x = x.max(dim=1, keepdim=True)[0]
        else:
            x = x.sum(dim=1, keepdim=True) / mask_rows.sum(dim=1, keepdim=True)

        x = self.mlp_theta_right_1(x)

        if torch.isnan(x).any():
            print('mlp')

        upper = torch.zeros(mask.size(0), self.k_eigval, self.k_eigval, device=mask.device, dtype=x.dtype)
        upper[torch.triu(torch.ones_like(upper), diagonal=1) == 1] = x.reshape(-1)
        skew = upper - torch.transpose(upper, -2, -1)            
        # Build rotation matrix from a skew-symetric matrix
        rot = torch.matrix_exp(skew)
        eigvec = eigvec @ rot
        eigvec = eigvec * mask_rows

        # Transformation
        x = torch.cat([inputs, eigvec], dim=-1) 
        x = self.mlp_between_rights(x)

        ## The second right mult pointnet
        x_r = self.pointnet_right_2(x, mask=mask_rows)

        x_r = x_r * mask_rows
        x_r = F.dropout(x_r, p=self.dropout, training=self.training)
        if self.max_pool:
            x_r = x_r.max(dim=1, keepdim=True)[0]
        else:
            x_r = x_r.sum(dim=1, keepdim=True) / mask_rows.sum(dim=1, keepdim=True)

        x_r = self.mlp_theta_right_2(x_r)

        if torch.isnan(x).any():
            print('mlp')

        upper = torch.zeros(x.size(0), x.size(-1), x.size(-1), device=mask.device, dtype=x.dtype)
        upper[torch.triu(torch.ones_like(upper), diagonal=1) == 1] = x_r.reshape(-1)
        skew = upper - torch.transpose(upper, -2, -1)
        # Build rotation matrix from a skew-symetric matrix
        rot = torch.matrix_exp(skew)
        x = x @ rot
        x = x * mask_rows


        # Read-out using PointNet
        x = self.pointnet_readout(x, mask=mask_rows)
        x = x * mask_rows
        x = F.dropout(x, p=self.dropout, training=self.training)
        if self.max_pool:
            x = x.max(dim=1, keepdim=False)[0]
        else:
            x = x.sum(dim=1, keepdim=False) / mask_rows.sum(dim=1, keepdim=False)

        x = self.mlp_out(x)
        if x.isnan().any():
            print('x_last', x)
        return x
