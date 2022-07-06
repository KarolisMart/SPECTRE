# Based on https://github.com/cvignac/SMP/blob/master/models/ppgn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from util.model_helper import masked_instance_norm2D, masked_layer_norm2D

SLOPE = 0.01

class PowerfulLayer(nn.Module):
    def __init__(self, in_feat: int, out_feat: int, num_layers: int, activation=nn.LeakyReLU(negative_slope=SLOPE), spectral_norm=(lambda x: x)):
        super().__init__()
        self.in_feat = in_feat
        self.out_feat= out_feat
        self.m1 =  nn.Sequential(*[spectral_norm(nn.Linear(in_feat if i ==0 else out_feat, out_feat)) if i % 2 == 0 else activation for i in range(num_layers*2-1)])
        self.m2 =  nn.Sequential(*[spectral_norm(nn.Linear(in_feat if i ==0 else out_feat, out_feat)) if i % 2 == 0 else activation for i in range(num_layers*2-1)])
        self.m4 = nn.Sequential(spectral_norm(nn.Linear(in_feat + out_feat, out_feat, bias=True)))

    def forward(self, x, mask):
        """
        x: batch x N x N x in_feat
        mask: batch x N x N x 1
        out: batch x N x N x out_feat
        """ 
        norm = mask[:,0].squeeze(-1).float().sum(-1).sqrt().view(mask.size(0), 1, 1, 1)
        mask = mask.unsqueeze(1).squeeze(-1)
        out1 = self.m1(x).permute(0, 3, 1, 2) * mask           # batch, out_feat, N, N
        out2 = self.m2(x).permute(0, 3, 1, 2) * mask           # batch, out_feat, N, N
        out = out1 @ out2                                      # batch, out_feat, N, N
        del out1, out2
        out =  out / norm                                      # Normalize to make std~1 (to preserve gradient norm, similar to self attention normalization)
        out = torch.cat((out.permute(0, 2, 3, 1), x), dim=3)   # batch, N, N, out_feat
        del x
        out = self.m4(out)
        return out

class FeatureExtractor(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation=nn.LeakyReLU(negative_slope=SLOPE), spectral_norm=(lambda x: x)):
        super().__init__()
        self.lin1 = nn.Sequential(spectral_norm(nn.Linear(in_features, out_features, bias=True)))
        self.lin2 = nn.Sequential(spectral_norm(nn.Linear(in_features, out_features, bias=False)))
        self.lin3 = nn.Sequential(spectral_norm(nn.Linear(out_features, out_features, bias=False)))
        self.activation = activation

    def forward(self, u, mask):
        """
        u: batch x N x N x in_feat
        mask: batch x N x N x 1
        out: batch x out_features
        """
        u = u * mask
        n = mask[:,0].sum(1)
        diag = u.diagonal(dim1=1, dim2=2)       # batch_size, channels, N
        trace = torch.sum(diag, dim=2)
        del diag
        out1 = self.lin1.forward(trace / n)

        s = (torch.sum(u, dim=[1, 2]) - trace) / (n * (n-1))
        del trace
        out2 = self.lin2.forward(s)             # bs, out_feat
        del s
        out = out1 + out2
        out = out + self.lin3.forward(self.activation(out))
        return out

class Powerful(nn.Module):
    def __init__(self, num_layers: int, input_features: int, hidden: int, hidden_final: int, dropout_prob: float,
                 simplified: bool, n_nodes: int, normalization: str = 'none', adj_out: bool = False,
                 output_features: int = 1, residual: bool = False, activation=nn.LeakyReLU(negative_slope=SLOPE),
                 spectral_norm=(lambda x: x), node_out: bool = False, node_output_features: int = 1):
        super().__init__()
        self.normalization = normalization
        layers_per_conv = 2 # was 1 originally, 2 worked better
        self.layer_after_conv = not simplified
        self.dropout_prob = dropout_prob
        self.adj_out = adj_out
        self.residual = residual
        self.activation = activation
        self.node_out = node_out

        self.no_prop = FeatureExtractor(hidden, hidden_final, activation=self.activation, spectral_norm=spectral_norm)
        self.in_lin = nn.Sequential(spectral_norm(nn.Linear(input_features, hidden)))
        self.layer_cat_lin = nn.Sequential(spectral_norm(nn.Linear(hidden*(num_layers+1), hidden)))
        self.convs = nn.ModuleList([])
        self.bns = nn.ModuleList([])
        for i in range(num_layers):
            self.convs.append(PowerfulLayer(hidden, hidden, layers_per_conv, activation=self.activation, spectral_norm=spectral_norm))

        self.feature_extractors = torch.nn.ModuleList([])
        for i in range(num_layers):
            if self.normalization == 'layer':
                self.bns.append(nn.LayerNorm([n_nodes,n_nodes,hidden], elementwise_affine=False))
            elif self.normalization == 'batch':
                self.bns.append(nn.BatchNorm2d(hidden))
            else:
                self.bns.append(None)
            self.feature_extractors.append(FeatureExtractor(hidden, hidden_final, activation=self.activation, spectral_norm=spectral_norm))
        if self.layer_after_conv:
            self.after_conv = nn.Sequential(spectral_norm(nn.Linear(hidden_final, hidden_final)))
        self.final_lin = nn.Sequential(spectral_norm(nn.Linear(hidden_final, output_features)))

        if self.node_out:
            self.layer_cat_lin_node = nn.Sequential(spectral_norm(nn.Linear(hidden*(num_layers+1), hidden)))
            if self.layer_after_conv:
                self.after_conv_node = nn.Sequential(spectral_norm(nn.Linear(hidden_final, hidden_final)))
            self.final_lin_node = nn.Sequential(spectral_norm(nn.Linear(hidden_final, node_output_features)))


    def forward(self, A, node_features, mask):
        # We do a concatenation of individual layer outputs, which is different from orginal (this seems worked better)
        mask = mask[..., None]  # batch, N, N, 1
        if len(A.shape) < 4:
            u = A[..., None]    # batch, N, N, 1
        else:
            u = A
        del A
        u = torch.cat([u, torch.diag_embed(node_features.transpose(-2,-1), dim1=1, dim2=2)], dim=-1)
        del node_features
        # if edge_features is not None:
        #     u[:,:,:,-edge_features.size(-1):][:, ~torch.eye(mask.size(1), dtype=torch.bool)] = edge_features
        # del edge_features
        u = u * mask
        u = self.in_lin(u)
        out = [u]
        for conv, extractor, bn in zip(self.convs, self.feature_extractors, self.bns):
            u = conv(u, mask) + (u if self.residual else 0)
            if self.normalization == 'none':
                u = u
            elif self.normalization == 'instance':
                u = masked_instance_norm2D(u, mask) # Didn't test it, but GraphNorm (https://arxiv.org/abs/2009.03294) might work even better
            elif self.normalization == 'layer':
                u = masked_layer_norm2D(u, mask)
            elif self.normalization == 'batch':
                u = bn(u.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            else:
                raise ValueError
            u = u * mask # Unnecessary with instance norm
            out.append(u)
        del u
        out = torch.cat(out, dim=-1)
        if self.node_out and self.adj_out:
            node_out = self.layer_cat_lin_node(out.diagonal(dim1=1, dim2=2).transpose(-2,-1))
            if self.layer_after_conv:
                node_out = node_out + self.activation(self.after_conv_node(node_out))
            node_out = F.dropout(node_out, p=self.dropout_prob, training=self.training)
            node_out = self.final_lin_node(node_out)
        out = self.layer_cat_lin(out)
        if not self.adj_out:
            out = self.feature_extractors[-1](out, mask)
        if self.layer_after_conv:
            out = out + self.activation(self.after_conv(out))
        out = F.dropout(out, p=self.dropout_prob, training=self.training)
        out = self.final_lin(out)
        if self.node_out and self.adj_out:
            return out, node_out
        else:
            return out
    