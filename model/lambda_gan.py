import torch
from torch import nn
import torch.nn.functional as F

class Transpose1dLayer(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        stride,
        padding=11,
        alpha=0.2,
        upsample=None,
        upsamping_mode='nearest',
        output_padding=1,
        gating=False,
        use_spectral_norm=False,
        activation='LeakyReLU',
        norm='none',
        out_len = 36,
    ):
        super(Transpose1dLayer, self).__init__()        

        self.upsample = upsample
        self.upsamping_mode = upsamping_mode
        self.alpha = alpha
        self.activation = activation
        self.norm = norm

        if 'batch' in self.norm :
            self.norm_l = nn.BatchNorm1d(input_channels)
        elif self.norm == 'layer':
            self.norm_l = nn.LayerNorm([input_channels, out_len//2])
        elif self.norm == 'instance':
            self.norm_l = nn.InstanceNorm1d(input_channels)

        self.gating = gating
        if self.gating:
            extra_channels = 2
        else:
            extra_channels = 1
        
        if self.upsample:
            reflection_pad = nn.ConstantPad1d(kernel_size // 2, value=0)
            conv1d = nn.Conv1d(input_channels, output_channels * extra_channels, kernel_size, stride)
            conv1d.weight.data.normal_(0.0, 0.02)
            if use_spectral_norm:
                conv1d = nn.utils.spectral_norm(conv1d)
            operation_list = [reflection_pad, conv1d]
        else:
            conv1dTrans = nn.ConvTranspose1d(input_channels, output_channels * extra_channels, kernel_size, stride, padding, output_padding)
            if use_spectral_norm:
                conv1dTrans = nn.utils.spectral_norm(conv1dTrans)
            operation_list = [conv1dTrans]
        
        self.transpose_ops = nn.Sequential(*operation_list)

    def forward(self, x, mask):
        if 'batch' in self.norm :
            x = self.norm_l(x)
        elif self.norm == 'layer':
            x = self.norm_l(x)
        elif self.norm == 'instance':
            x = self.norm_l(x)
            
        if self.upsample:
            # recommended by wavgan paper to use nearest upsampling, replace by nn.Upsample layer for cleanliness?
            x = nn.functional.interpolate(x, scale_factor=self.upsample, mode=self.upsamping_mode)
        x = self.transpose_ops(x)
        if self.gating and self.activation == 'tanh_gated_last':
            x = 2*(torch.tanh(x[:,:(x.size(1)//2),:])) * torch.sigmoid(x[:,(x.size(1)//2):,:])
        elif self.gating :
            x = torch.tanh(x[:,:(x.size(1)//2),:]) * torch.sigmoid(x[:,(x.size(1)//2):,:])
        elif self.activation == 'tanh':
            x = torch.tanh(x)
        elif self.activation == 'tanh_last':
            x = 2*torch.tanh(x) # Ensure activation covers the range of [0, 2]
            # x = torch.tanh(x) + 1
        elif self.activation == 'linear':
            x = x
        else:
            x = F.gelu(x)
        
        return x


class Conv1D(nn.Module):
    def __init__(
        self,
        input_channels,
        output_channels,
        kernel_size,
        alpha=0.2,
        shift_factor=2,
        stride=4,
        padding=11,
        drop_prob=0,
        gating=False,
        use_spectral_norm=False,
        dilation=1,
        norm='none',
        out_len = 36
    ):
        super(Conv1D, self).__init__()
        
        self.alpha = alpha
        self.use_phase_shuffle = (shift_factor > 0)
        self.use_dropout = drop_prob > 0
        self.norm = norm

        if 'layer' in self.norm:
            self.norm_l = nn.LayerNorm([input_channels, out_len * 2])
        elif 'instance' in self.norm:
            self.norm_l = nn.InstanceNorm1d(input_channels)

        self.gating = gating
        if self.gating:
            extra_channels = 2
        else:
            extra_channels = 1

        self.conv1d = nn.Conv1d(input_channels, output_channels * extra_channels, kernel_size, stride=stride, padding=padding, dilation=dilation)
        if use_spectral_norm:
            self.conv1d = nn.utils.spectral_norm(self.conv1d)
        self.dropout = nn.Dropout2d(drop_prob)


    def forward(self, x, mask):
        if self.norm == 'layer':
            x = self.norm_l(x)
        elif self.norm == 'instance':
            x = self.norm_l(x)
        x = self.conv1d(x)
        if self.gating:
            x = torch.tanh(x[:,:(x.size(1)//2),:]) * torch.sigmoid(x[:,(x.size(1)//2):,:])
        else:
            x = F.gelu(x)
        if self.use_dropout:
            x = self.dropout(x)
        
        return x

class LambdaGenerator(nn.Module):
    def __init__(
        self,
        alpha=0.2,
        data_channels=128,
        noise_latent_dim=100,
        gelu=False,
        k_eigval=18,
        dropout=0.0,
        n_max=36,
        norm='instance',
        upsample=False,
        upsamping_mode='nearest',
        gating=False,
        last_gating=False,
        last_linear=False,
        fixed_init=False,
    ):
        super(LambdaGenerator, self).__init__()
        
        self.data_channels = data_channels
        self.noise_latent_dim = noise_latent_dim
        self.k_eigval = k_eigval
        self.alpha = alpha
        self.n_max = n_max
        self.dropout = dropout
        self.fixed_init = fixed_init

        if self.fixed_init:
            self.init_lambda = nn.Parameter(torch.Tensor(self.k_eigval))
            self.init_lambda.requires_grad = True
            nn.init.constant_(self.init_lambda, 2.0/n_max)

        self.gating = gating

        if gelu:
            activation = nn.GELU()
        else:
            activation = nn.LeakyReLU(negative_slope=alpha)
        self.activation = activation

        self.data_channels = self.data_channels
        input_features = noise_latent_dim
        if not self.gating:
            self.data_channels = self.data_channels * 2

        if self.k_eigval > 32:
            self.data_channels = self.data_channels * 2 
            if self.k_eigval > 64:
                self.data_channels = self.data_channels * 2 

        fc1_out_featrues = 2 * self.data_channels
        self.in_size = 2

        self.fc1 = nn.Sequential(
                                    nn.Linear(input_features, fc1_out_featrues),
                                    activation,
                                )
        
        last_alpha = alpha
        if last_gating:
            last_activation = 'tanh_gated_last'
        elif last_linear:
            last_activation = 'linear'
        else:
            last_activation = 'tanh_last'
        
        stride = 2
        if upsample:
            stride = 1
            upsample = 2

        deconv_layers = [
            Transpose1dLayer(
                self.data_channels,
                (self.data_channels // 2),
                5,
                stride,
                upsample=upsample,
                upsamping_mode=upsamping_mode,
                alpha=alpha,
                padding=2,
                gating=self.gating,
                use_spectral_norm=False,
                norm=norm,
                out_len=4
            ),
            Transpose1dLayer(
                self.data_channels // 2,
                (self.data_channels // 4),
                9,
                stride,
                upsample=upsample,
                upsamping_mode=upsamping_mode,
                alpha=alpha,
                padding=4,
                gating=self.gating,
                use_spectral_norm=False,
                norm=norm,
                out_len=8
            ),
            Transpose1dLayer(
                self.data_channels // 4,
                (self.data_channels // 8),
                17,
                stride,
                upsample=upsample,
                upsamping_mode=upsamping_mode,
                alpha=alpha,
                padding=8,
                gating=self.gating,
                use_spectral_norm=False,
                norm=norm,
                out_len=16
            ),
            Transpose1dLayer(
                self.data_channels // 8,
                ((self.data_channels // 16) if self.k_eigval > 32 else 1),
                25,
                stride,
                upsample=upsample,
                upsamping_mode=upsamping_mode,
                alpha=last_alpha,
                padding=12,
                gating=(self.gating if self.k_eigval > 32 else last_gating),
                use_spectral_norm=False,
                activation=last_activation,
                norm=norm,
                out_len=32
            )
        ]
        if self.k_eigval > 64:
            deconv_layers += [
                Transpose1dLayer(
                    self.data_channels // 16,
                    self.data_channels // 32,
                    25,
                    stride,
                    upsample=upsample,
                    upsamping_mode=upsamping_mode,
                    alpha=last_alpha,
                    padding=12,
                    gating=self.gating,
                    use_spectral_norm=False,
                    activation=last_activation,
                    norm=norm,
                    out_len=32
                ),
                Transpose1dLayer(
                    self.data_channels // 32,
                    1,
                    25,
                    stride,
                    upsample=upsample,
                    upsamping_mode=upsamping_mode,
                    alpha=last_alpha,
                    padding=12,
                    gating=last_gating,
                    use_spectral_norm=False,
                    activation=last_activation,
                    norm=norm,
                    out_len=32
                )
            ]
        elif self.k_eigval > 32:
            deconv_layers += [
                Transpose1dLayer(
                    self.data_channels // 16,
                    1,
                    25,
                    stride,
                    upsample=upsample,
                    upsamping_mode=upsamping_mode,
                    alpha=last_alpha,
                    padding=12,
                    gating=last_gating,
                    use_spectral_norm=False,
                    activation=last_activation,
                    norm=norm,
                    out_len=32
                )
            ]

        self.deconv_list = nn.ModuleList(deconv_layers)
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                if gelu:
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                else:
                    nn.init.kaiming_uniform_(m.weight.data, a=self.alpha)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, noise, mask):
        mask_eigvals = mask[:,0,:self.k_eigval]
        mask = mask[:,0]
        x = noise
        x = self.fc1(x)
        x = x.view(mask.size(0), self.data_channels, self.in_size)
        del noise
        x = F.dropout(x, p=self.dropout, training=self.training)
        for i, deconv in enumerate(self.deconv_list):
            x = deconv(x, mask)
        x = x.squeeze(1)
        x = x[:,:self.k_eigval]
        if self.fixed_init:
            x = x + self.init_lambda
        x[:,:self.k_eigval] = x[:,:self.k_eigval] * mask_eigvals
        return x

class LambdaDiscriminator(nn.Module):
    def __init__(
        self,
        alpha=0.2,
        data_channels=128,
        use_spectral_norm=False,
        gelu=False,
        k_eigval=18,
        dropout=0.0,
        n_max=36,
        norm='instance',
        gating=False,
    ):
        super(LambdaDiscriminator, self).__init__()
        self.alpha = alpha
        self.data_channels = data_channels
        self.k_eigval = k_eigval
        self.dropout = dropout
        self.n_max = n_max

        if gelu:
            activation = nn.GELU()
        else:
            activation = nn.LeakyReLU(negative_slope=alpha)
        self.activation = activation

        num_channels = 2 # n, eigvals

        self.gating = gating # WAS True - just for test

        self.data_channels = self.data_channels

        if not self.gating:
            self.data_channels = self.data_channels * 2

        if self.k_eigval > 32:
            self.data_channels = self.data_channels * 2 
            if self.k_eigval > 64:
                self.data_channels = self.data_channels * 2 

        shift_factor=0
        stride = 2
        conv_layers = []
        if self.k_eigval > 64:
            conv_layers += [
                Conv1D(
                    num_channels,
                    self.data_channels // 32,
                    25,
                    stride=stride,
                    padding=12,
                    alpha=alpha,
                    shift_factor=shift_factor,
                    gating=self.gating,
                    use_spectral_norm=use_spectral_norm,
                    norm='none',
                    out_len=16
                ),
                Conv1D(
                    self.data_channels // 32,
                    self.data_channels // 16,
                    25,
                    stride=stride,
                    padding=8,
                    alpha=alpha,
                    shift_factor=shift_factor,
                    gating=self.gating,
                    use_spectral_norm=use_spectral_norm,
                    norm=norm,
                    out_len=8
                ),
            ]
        elif self.k_eigval > 32:
            conv_layers += [
                Conv1D(
                    num_channels,
                    self.data_channels // 16,
                    25,
                    stride=stride,
                    padding=12,
                    alpha=alpha,
                    shift_factor=shift_factor,
                    gating=self.gating,
                    use_spectral_norm=use_spectral_norm,
                    norm='none',
                    out_len=16
                ),
            ]
        conv_layers += [
            Conv1D(
                (self.data_channels // 16 if self.k_eigval > 32 else num_channels),
                self.data_channels // 8,
                25,
                stride=stride,
                padding=12,
                alpha=alpha,
                shift_factor=shift_factor,
                gating=self.gating,
                use_spectral_norm=use_spectral_norm,
                norm='none',
                out_len=16
            ),
            Conv1D(
                self.data_channels // 8,
                self.data_channels // 4,
                17,
                stride=stride,
                padding=8,
                alpha=alpha,
                shift_factor=shift_factor,
                gating=self.gating,
                use_spectral_norm=use_spectral_norm,
                norm=norm,
                out_len=8
            ),
            Conv1D(
                self.data_channels // 4,
                self.data_channels // 2,
                9,
                stride=stride,
                padding=4,
                alpha=alpha,
                shift_factor=shift_factor,
                gating=self.gating,
                use_spectral_norm=use_spectral_norm,
                norm=norm,
                out_len=4
            ),
            Conv1D(
                self.data_channels // 2,
                self.data_channels,
                5,
                stride=stride,
                padding=2,
                alpha=alpha,
                shift_factor=shift_factor,
                gating=self.gating,
                use_spectral_norm=use_spectral_norm,
                norm=norm, 
                out_len=2
            )
        ]
       
        self.fc_input_size = 2 * self.data_channels
        self.conv_layers = nn.ModuleList(conv_layers)

        self.in_len = 32

        if use_spectral_norm:
            self.fc1 = nn.Sequential((nn.Identity() if 'none' in norm else nn.LayerNorm(self.fc_input_size)), nn.utils.spectral_norm(nn.Linear(self.fc_input_size, 1)))
        else:
            self.fc1 = nn.Sequential((nn.Identity() if 'none' in norm else nn.LayerNorm(self.fc_input_size)), nn.Linear(self.fc_input_size, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                if gelu:
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                else:
                    nn.init.kaiming_uniform_(m.weight.data, a=self.alpha)
                if m.bias is not None:
                    nn.init.zeros_(m.bias.data)

    def forward(self, eigval, mask):
        mask = mask[:,0]
        n = torch.sum(mask, dim=-1, keepdim=True)
        n = n/self.n_max
        mask_eigvals = mask[:,:self.k_eigval]
        eigval = eigval[:,:self.k_eigval] * mask_eigvals
        eigval = eigval * mask_eigvals
        eigval = eigval.reshape(mask.size(0), 1, -1)
        size_diff = self.in_len - eigval.size(-1)
        if size_diff < 0:
            eigval = eigval[:,:self.in_len]
        else:
            eigval = F.pad(eigval, [0, size_diff])
        x = torch.cat([eigval, n.unsqueeze(-1).expand(-1, -1, eigval.size(-1))], dim=1)
        for conv in self.conv_layers:
            x = conv(x, mask)
        x = x.view(mask.size(0), self.fc_input_size)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.fc1(x)
