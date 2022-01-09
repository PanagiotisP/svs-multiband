# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Created on 2018/12
# Author: Kaituo XU
# Modified on 2019/11 by Alexandre Defossez, added support for multiple output channels
# Here is the original license:
# The MIT License (MIT)
#
# Copyright (c) 2018 Kaituo XU
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .cls_basic_conv1ds import ConvEncoder 
from .cls_fe_nnct import Synthesis 
from .gammatone import generate_mpgtf
from .utils import capture_init, center_trim

EPS = 1e-8

x = True

def overlap_and_add(signal, frame_step):
    outer_dimensions = signal.size()[:-2]
    frames, frame_length = signal.size()[-2:]

    subframe_length = math.gcd(frame_length, frame_step)  # gcd=Greatest Common Divisor
    subframe_step = frame_step // subframe_length
    subframes_per_frame = frame_length // subframe_length
    output_size = frame_step * (frames - 1) + frame_length
    output_subframes = output_size // subframe_length

    subframe_signal = signal.view(*outer_dimensions, -1, subframe_length)

    frame = torch.arange(0, output_subframes,
                         device=signal.device).unfold(0, subframes_per_frame, subframe_step)
    frame = frame.long()  # signal may in GPU or CPU
    frame = frame.contiguous().view(-1)

    result = signal.new_zeros(*outer_dimensions, output_subframes, subframe_length)
    result.index_add_(-2, frame, subframe_signal)
    result = result.view(*outer_dimensions, -1)
    return result


class SeqConvTasNet(nn.Module):
    @capture_init
    def __init__(self,
                 enc_dim=256,
                 transform_window=20,
                 bottle_dim=256,
                 hidden_dim=512,
                 kernel_size=3,
                 stacks=9,
                 racks=3,
                 instr_num=2,
                 audio_channels=2,
                 norm_type="gLN",
                 pad=False,
                 band_num=1,
                 skip=False,
                 dwt=False,
                 dwt_time=True,
                 learnable='none',
                 samplerate=22050,
                 stronger_enc=False,
                 concat_features=True):
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            pad: use of padding in TCN or not
            band_num: Number of frequenyc bands to split processing into
            skip: determines TCN output, skip or residual
            learnable: 'none', 'normal', 'inverse', 'both'
            determines learnable initial transformation layer to be paired with dwt
        """
        super(SeqConvTasNet, self).__init__()

        # Hyper-parameter
        self.instr_num = instr_num
        self.learnable = learnable
        self.dwt_time = dwt_time
        self.samplerate = samplerate
        # Components
        learnable_transform = nn.Sequential(
            nn.Conv1d(audio_channels, audio_channels * 2, kernel_size, padding=1, bias=False),
            nn.AvgPool1d(2, padding=1))
        learnable_inverse_transform = nn.ConvTranspose1d(audio_channels * 2,
                                                         audio_channels,
                                                         kernel_size,
                                                         padding=1,
                                                         stride=2,
                                                         bias=False,
                                                         output_padding=1)
        if self.learnable == 'normal':
            self.learnable_transform = learnable_transform
        elif self.learnable == 'inverse':
            self.learnable_inverse_transform = learnable_inverse_transform
        elif self.learnable == 'both':
            self.learnable_transform = learnable_transform
            self.learnable_inverse_transform = learnable_inverse_transform
        if self.dwt_time:
            self.dwt_layer = DWaveletTransformation()

        self.conv_tasnet1 = ConvTasNet(audio_channels=audio_channels,
                                       stacks=stacks,
                                       pad=pad,
                                       band_num=band_num,
                                       skip=skip,
                                       racks=racks,
                                       hidden_dim=hidden_dim,
                                       bottle_dim=bottle_dim,
                                       enc_dim=enc_dim,
                                       instr_num=instr_num,
                                       transform_window=transform_window,
                                       norm_type=norm_type,
                                       dwt=dwt,
                                       stronger_enc=stronger_enc,
                                       concat_features=concat_features)
        self.conv_tasnet2 = ConvTasNet(audio_channels=audio_channels,
                                       stacks=stacks,
                                       pad=pad,
                                       band_num=band_num,
                                       skip=skip,
                                       racks=racks,
                                       hidden_dim=hidden_dim,
                                       bottle_dim=bottle_dim,
                                       enc_dim=enc_dim,
                                       instr_num=instr_num,
                                       transform_window=transform_window,
                                       norm_type=norm_type,
                                       dwt=dwt,
                                       stronger_enc=stronger_enc,
                                       concat_features=concat_features)

    def valid_length(self, length):
        return self.conv_tasnet2.valid_length(length)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
        if self.dwt_time:
            # Pass input from transformation
            if self.learnable != 'normal' and self.learnable != 'both':
                dwt_output = self.dwt_layer(mixture, False)
            else:
                dwt_output = self.learnable_transform(mixture)

            # Split channels by half
            low = dwt_output[:, :mixture.shape[1]]
            high = dwt_output[:, mixture.shape[1]:]

            # Process each half individually
            est_source_1 = self.conv_tasnet1(low)
            est_source_2 = self.conv_tasnet2(high)

            # Pass input from inverse transformation
            clean_signals = []
            if self.learnable != 'inverse' and self.learnable != 'both':
                for instr_num in range(self.instr_num):
                    combined_signal = torch.cat(
                        [est_source_1[:, instr_num], est_source_2[:, instr_num]], dim=1)
                    clean_signals.append(self.dwt_layer(combined_signal, True))
            else:
                for instr_num in range(self.instr_num):
                    combined_signal = torch.cat(
                        [est_source_1[:, instr_num], est_source_2[:, instr_num]], dim=1)
                    clean_signals.append(self.learnable_inverse_transform(combined_signal, True))

            est_source = torch.stack(clean_signals, dim=1)
        return est_source


class ConvTasNet(nn.Module):
    @capture_init
    def __init__(self,
                 enc_dim=256,
                 transform_window=20,
                 bottle_dim=256,
                 hidden_dim=512,
                 kernel_size=3,
                 stacks=8,
                 racks=3,
                 instr_num=2,
                 audio_channels=2,
                 norm_type="gLN",
                 pad=False,
                 band_num=1,
                 dilation_split=False,
                 skip=False,
                 dwt=False,
                 samplerate=22050,
                 stronger_enc=False,
                 concat_features=True,
                 full_band=False,
                 sinc_conv=False,
                 gammatone=False,
                 transposed_decoder=False,
                 post_enc_linear=False,
                 phase_split_filterbank=False,
                 frozen_unsupervised_enc_dec=False,
                 frozen_waveunet_enc_dec=False,
                 frozen_original_enc_dec=False):
        """
        Args:
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            audio_channels: Number of audio channels (1 for mono, 2 for stereo)
            norm_type: BN, gLN, cLN
            pad: use of padding in TCN or not
            band_num: how many frequency bands to process. More practically, this is the number
            of parallel separator/TCN modules
            dilation_split: separate layers within one rack
            skip: determines TCN output, skip or residual
        """
        super(ConvTasNet, self).__init__()
        assert enc_dim % band_num == 0, 'Encoding dimension must be divisible by band num'
        assert not dilation_split or stacks % band_num == 0, 'Dilation split requires stacks being divisible by band num'
        # Hyper-parameter
        self.transform_window = transform_window
        self.kernel_size = kernel_size
        self.audio_channels = audio_channels
        self.stacks = stacks
        self.racks = racks
        self.pad = pad
        self.band_num = band_num
        self.dilation_split = dilation_split
        self.dwt = dwt
        self.even_pad = None
        self.samplerate = samplerate
        self.full_band = full_band
        self.sinc_conv = sinc_conv
        self.post_enc_linear = post_enc_linear
        self.frozen_waveunet_enc_dec = frozen_waveunet_enc_dec
        # Components
        if stronger_enc:
            self.encoder = StrongerEncoder(
                                   transform_window,
                                   enc_dim,
                                   layers=6,
                                   num_mels=256,
                                   audio_channels=audio_channels,
                                   samplerate=samplerate,
                                   concat_features=concat_features)
        elif sinc_conv:
            self.encoder = SincConv_fast(enc_dim, transform_window, sample_rate=samplerate, in_channels=audio_channels, stride=transform_window//2)
        elif gammatone:
            self.encoder = GammatoneEncoder(enc_dim, transform_window, transform_window//2, sample_rate=samplerate, in_channels=audio_channels, phase_split=phase_split_filterbank)
        elif frozen_unsupervised_enc_dec:
            channels = 800
            ft_size = 2048
            ft_size_space = 800
            hop_size = 256
            pretrained_weights = torch.load('pretrained/analysis.python', map_location='cpu')
            analysis = cls_basic_conv1ds.ConvEncoder(in_size=ft_size,
                                                 out_size=ft_size_space,
                                                 hop_size=hop_size, exp_settings={'fully_modulated': True, 'monaural': False, 'fs':22050, 'dp_length': 2})
            self.encoder = nn.Sequential([analysis])
        elif frozen_waveunet_enc_dec:
            encoder = nn.Conv1d(1, enc_dim, transform_window, stride=transform_window//2)
            bias = nn.Parameter(torch.from_numpy(np.load('pretrained_conv_fbanks/enc_biases.npz')['arr_0']),
                                requires_grad=False)
            weight = nn.Parameter(torch.from_numpy(np.transpose(np.load('pretrained_conv_fbanks/enc_weights.npz')['arr_0'], (2,1,0))), requires_grad=False)
            encoder.weight = weight
            encoder.bias = bias
            self.encoder = nn.Sequential(nn.Conv1d(audio_channels, 1, 1, bias=False), encoder)
        elif frozen_original_enc_dec:
            self.encoder = Encoder(transform_window, enc_dim,
                               audio_channels)
            self.encoder.load_state_dict(torch.load('pretrained_conv_fbanks/encoder'))
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False
        else:
            self.encoder = Encoder(transform_window, enc_dim,
                               audio_channels) 
        if dwt:
            self.dwt_layer = DWaveletTransformation()

        if post_enc_linear:
            self.enc_linear = nn.Linear(enc_dim, enc_dim, bias=False)
            
        self.separators = nn.ModuleList()
        if dilation_split:
            for i in range(self.band_num):
                self.separators.append(
                    TemporalConvNet(enc_dim,
                                    bottle_dim,
                                    hidden_dim,
                                    kernel_size,
                                    stacks // self.band_num,
                                    racks,
                                    instr_num if i == (self.band_num - 1) else 1,
                                    norm_type,
                                    pad,
                                    dilation_group=i,
                                    skip=skip))
        elif band_num >= 1:
            total_dim = 0
            for i in range(band_num):
                total_dim += enc_dim // band_num
                self.separators.append(
                    TemporalConvNet(enc_dim // band_num,
                                    bottle_dim,
                                    hidden_dim,
                                    kernel_size,
                                    stacks,
                                    racks,
                                    instr_num,
                                    norm_type,
                                    pad,
                                    skip=skip))
            if full_band:
                self.separators.append(
                    TemporalConvNet(enc_dim,
                                    bottle_dim,
                                    hidden_dim,
                                    kernel_size,
                                    stacks,
                                    racks,
                                    instr_num,
                                    norm_type,
                                    pad,
                                    skip=skip))
                self.linear_mask_combinator = nn.Linear(2*enc_dim, enc_dim, bias=False)
        else:
            self.separators.append(
                TemporalConvNet(enc_dim,
                                bottle_dim,
                                hidden_dim,
                                kernel_size,
                                stacks,
                                racks,
                                instr_num,
                                norm_type,
                                pad,
                                skip=skip))

#        if post_enc_linear:
#            self.dec_linear = nn.Linear(enc_dim, enc_dim, bias=False)

        if frozen_unsupervised_enc_dec:
            synthesis = cls_fe_nnct.Synthesis(ft_size=exp_settings['ft_size_space'],
                                              kernel_size=exp_settings['ft_syn_size'],
                                              hop_size=exp_settings['hop_size'], exp_settings=exp_settings)

        elif frozen_waveunet_enc_dec:
            decoder = nn.ConvTranspose1d(enc_dim, 1, transform_window, stride=transform_window//2, bias=False)
#             bias = nn.Parameter(torch.from_numpy(np.load('pretrained_conv_fbanks/dec_biases.npz')['arr_0']),
#                                 requires_grad=False)
            weight = nn.Parameter(torch.from_numpy(np.transpose(np.load('pretrained_conv_fbanks/dec_weights.npz')['arr_0'], (3,1,2,0))).squeeze(1), requires_grad=False)
            decoder.weight = weight
            self.decoder = nn.Sequential(decoder, nn.Conv1d(1, audio_channels, 1, bias=False))
            
        elif stronger_enc:
            self.decoder = StrongerDecoder(
                                   transform_window,
                                   enc_dim,
                                   layers=6,
                                   audio_channels=audio_channels)
        elif frozen_original_enc_dec:
            self.decoder = Decoder(enc_dim, transform_window,
                               audio_channels, stride=transform_window//2, transposed_decoder=transposed_decoder)
            self.decoder.load_state_dict(torch.load('pretrained_conv_fbanks/decoder'))
            for parameter in self.decoder.parameters():
                parameter.requires_grad = False
        
        else:
            self.decoder = Decoder(enc_dim, transform_window,
                               audio_channels, stride=transform_window//2, transposed_decoder=transposed_decoder)
        # init
        for p in self.parameters():
            if p.dim() > 1 and p.requires_grad == True:
                nn.init.xavier_normal_(p)

    def valid_length(self, length):
        if not self.pad:
            length += self.racks * (self.kernel_size - 1) * (2**self.stacks -
                                                             1) * self.transform_window // 2
        if length % self.transform_window // 2 != 0:
            length += (self.transform_window // 2) - (length % (self.transform_window // 2))
        if self.sinc_conv:
            length += 1
        return length

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            est_source: [M, C, T]
        """
#         print('mixture', mixture.shape)
#         global x
#         if x:
#             self.decoder.load_state_dict(torch.load('pretrained_conv_fbanks/decoder'))
#             x = False
#         print(self.encoder.conv1d_U.weight)
#         print(self.decoder.basis_signals.weight)
        mixture_w = self.encoder(mixture)
#         print('encoded', mixture_w.shape)
        if self.dwt:
            if mixture_w.shape[-1] % 2 != 0:
                mixture_w = F.pad(mixture_w, (0, 1))
                self.even_pad = True
            mixture_w = self.dwt_layer(mixture_w, inverse=False)
        if self.post_enc_linear:
            mixture_w = self.enc_linear(mixture_w.transpose(1,2)).transpose(1,2)
        split_mixture_w = torch.chunk(mixture_w, self.band_num, dim=1)
        est_masks = []
        est_mask = 0.
        if self.dilation_split:
            tcn_output = mixture_w
            for i in range(self.band_num - 1):
                tcn_output = self.separators[i](tcn_output).squeeze(dim=1)
            est_mask = self.separators[self.band_num - 1](tcn_output)
        elif self.band_num > 1:
            for i in range(self.band_num):
                est_masks.append(self.separators[i](split_mixture_w[i]))
            if self.full_band:
                est_masks.append(self.separators[-1](mixture_w))
            est_mask = torch.cat(est_masks, dim=2)
            if self.full_band:
                est_mask = est_mask.transpose(2,3)
                est_mask = self.linear_mask_combinator(est_mask)
                est_mask = est_mask.transpose(2,3)
            
        else:
            # Basic model, only one separator
            est_mask = self.separators[0](mixture_w)
#         print('est_mask', est_mask.shape)  
        mixture_w = torch.unsqueeze(center_trim(mixture_w, est_mask), 1) * est_mask  # [M, C, N, K]
        if self.dwt:
            mixture_w = self.dwt_layer(mixture_w, inverse=True)
            if self.even_pad:
                mixture_w = mixture_w[:, :, :, :-1]
#        if self.post_enc_linear:
#            mixture_w = self.dec_linear(mixture_w.transpose(2,3)).transpose(2,3)
        if self.frozen_waveunet_enc_dec:
            batch_size, instr_num, enc_dim, feature_map_size = mixture_w.shape
            est_source = self.decoder(mixture_w.view(batch_size*instr_num, enc_dim, feature_map_size)).view(batch_size, instr_num, self.audio_channels, -1)
        else:
            est_source = self.decoder(mixture_w)
#         print('est_source', est_source.shape)

        
        return est_source


class Encoder(nn.Module):
    """Estimation of the nonnegative mixture weight by a 1-D conv layer.
    """
    def __init__(self, kernel_size, enc_dim, audio_channels):
        super(Encoder, self).__init__()
        # Components
        # 50% overlap
        self.conv1d_U = nn.Conv1d(audio_channels,
                                  enc_dim,
                                  kernel_size=kernel_size,
                                  stride=kernel_size // 2,
                                  bias=False)

    def forward(self, mixture):
        """
        Args:
            mixture: [M, T], M is batch size, T is #samples
        Returns:
            mixture_w: [M, N, K], where K = (T-L)/(L/2)+1 = 2T/L-1
        """
        mixture_w = F.relu(self.conv1d_U(mixture))  # [M, N, K]
        return mixture_w


class DWaveletTransformation(nn.Module):
    def __init__(self, p_scale=1, u_scale=0.5, a_scale=np.sqrt(2)):
        super(DWaveletTransformation, self).__init__()
        self.scales = {'p': p_scale, 'u': u_scale, 'a': a_scale}

    def forward(self, z, inverse):
        if not inverse:
            z_odd, z_even = z[:, :, 1::2], z[:, :, ::2]
            error = z_odd - self.scales['p'] * z_even
            signal = z_even + self.scales['u'] * error
            error = error / self.scales['a']
            signal = signal * self.scales['a']
            mixture_w = torch.cat((error, signal), 1)
            return mixture_w
        else:
            enc_dim = z.shape[2] // 2
            error, signal = z[:, :, :enc_dim, :], z[:, :, enc_dim:, :]
            signal = signal / self.scales['a']
            error = error * self.scales['a']
            z_even = signal - self.scales['u'] * error
            z_odd = error + self.scales['p'] * z_even

            source_w = torch.zeros(
                (z_even.shape[0], z_even.shape[1], z_even.shape[2], z_even.shape[3] * 2)).cuda()

            source_w[:, :, :, 1::2] = z_odd
            source_w[:, :, :, ::2] = z_even
            return source_w


class Decoder(nn.Module):
    def __init__(self, enc_dim, transform_window, audio_channels, stride=None, transposed_decoder=False, instr_num=2):
        super(Decoder, self).__init__()
        # Hyper-parameter
        self.transform_window = transform_window
        self.audio_channels = audio_channels
        self.transposed_decoder = transposed_decoder
        # Components
        if transposed_decoder:
            self.conv_transp = nn.ConvTranspose1d(enc_dim, audio_channels, transform_window,transform_window//2, bias=False)
        else:
            self.basis_signals = nn.Linear(enc_dim, audio_channels * transform_window, bias=False)
        self.stride = stride
    def forward(self, mixture_w):
        """
        Args:
            mixture_w: [M, N, K]
            est_mask: [M, C, N, K]
        Returns:
            est_source: [M, C, T]
        """
        # D = W * M
        # source_w = torch.unsqueeze(mixture_w, 1) * est_mask  # [M, C, N, K]
        batch_size, instr_num, enc_dim, feature_map_size = mixture_w.shape
        if self.transposed_decoder:
            est_source = self.conv_transp(mixture_w.view(batch_size*instr_num, enc_dim, feature_map_size)).view(batch_size, instr_num, self.audio_channels, -1)
        else:
            source_w = torch.transpose(mixture_w, 2, 3)  # [M, C, K, N]
            # S = DV
            est_source = self.basis_signals(source_w)  # [M, C, K, ac * L]
            m, c, k, _ = est_source.size()
            est_source = est_source.view(m, c, k, self.audio_channels, -1).transpose(2, 3).contiguous()
            est_source = overlap_and_add(est_source, self.stride or self.transform_window // 2)  # M x C x ac x T
        return est_source


class TemporalConvNet(nn.Module):
    def __init__(self,
                 enc_dim,
                 bottle_dim,
                 hidden_dim,
                 kernel_size,
                 stacks,
                 racks,
                 instr_num,
                 norm_type="gLN",
                 pad=False,
                 dilation_group=0,
                 skip=False):
        """
        Args:
            N: Number of filters in autoencoder
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            C: Number of speakers
            norm_type: BN, gLN, cLN
            pad: Pad or not
            dilation_group: Used in dilation split tests. Helps to define the starting dilation rate
        """
        super(TemporalConvNet, self).__init__()
        # Hyper-parameter
        self.instr_num = instr_num
        self.pad = pad
        self.skip = skip
        # Components
        # [M, N, K] -> [M, N, K]
        layer_norm = ChannelwiseLayerNorm(enc_dim)
        # [M, N, K] -> [M, B, K]
        bottleneck_conv1x1 = nn.Conv1d(enc_dim, bottle_dim, 1, bias=False)
        # [M, B, K] -> [M, B, K]
        repeats = []
        for _ in range(racks):
            blocks = []
            for x in range(stacks):
                dilation = 2**(x + dilation_group * stacks)
                padding = (kernel_size - 1) * dilation // 2
                padding = padding if pad else 0
                blocks += [
                    TemporalBlock(bottle_dim,
                                  hidden_dim,
                                  kernel_size,
                                  stride=1,
                                  padding=padding,
                                  dilation=dilation,
                                  norm_type=norm_type,
                                  pad=pad,
                                  skip=skip)
                ]
            repeats += [nn.ModuleList(blocks)]
        temporal_conv_net = nn.ModuleList(repeats)

        # [M, B, K] -> [M, C*N, K]
        mask_conv1x1 = nn.Conv1d(bottle_dim, instr_num * enc_dim, 1, bias=False)
        # Put together
        self.network = nn.Sequential(layer_norm, bottleneck_conv1x1)
        self.temporal_conv_net = temporal_conv_net
        self.mask_conv1x1 = mask_conv1x1

    def forward(self, mixture_w):
        """
        Keep this API same with TasNet
        Args:
            mixture_w: [M, N, K], M is batch size
        returns:
            est_mask: [M, C, N, K]
        """
        M, N, _ = mixture_w.size()
        score = self.network(mixture_w)  # [M, N, K] -> [M, C*N, K]
        skip_connection_sum = 0.
        residual = score
        for rack in self.temporal_conv_net:
            for block in rack:
                residual, skip = block(residual)
                if self.skip:
                    skip_connection_sum = skip + \
                        (skip_connection_sum if self.pad else center_trim(skip_connection_sum, skip))
        if self.skip:
            score = self.mask_conv1x1(skip_connection_sum)
        else:
            score = self.mask_conv1x1(residual)
        score = score.view(M, self.instr_num, N, -1)  # [M, C*N, K] -> [M, C, N, K]
        est_mask = F.relu(score)
        return est_mask


class TemporalBlock(nn.Module):
    def __init__(self,
                 bottle_dim,
                 hidden_dim,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 norm_type="gLN",
                 pad=False,
                 skip=False):
        super(TemporalBlock, self).__init__()
        self.pad = pad
        # [M, B, K] -> [M, H, K]
        conv1x1 = nn.Conv1d(bottle_dim, hidden_dim, 1, bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, hidden_dim)
        # [M, H, K] -> [M, B, K]
        dsconv = DepthwiseSeparableConv(hidden_dim,
                                        bottle_dim,
                                        kernel_size,
                                        stride,
                                        padding,
                                        dilation,
                                        norm_type,
                                        pad=pad,
                                        skip=skip)
        # Put together
        self.net = nn.Sequential(conv1x1, prelu, norm, dsconv)

    def forward(self, x):
        """
        Args:
            x: [M, B, K]
        Returns:
            [M, B, K]
        """
        residual = x
        # TODO: when P = 3 here works fine, but when P = 2 maybe need to pad?
        out, skip = self.net(x)
        # look like w/o F.relu is better than w/ F.relu
        return out + (residual if self.pad else center_trim(residual, out)), skip


class DepthwiseSeparableConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation,
                 norm_type="gLN",
                 pad=False,
                 skip=False):
        super(DepthwiseSeparableConv, self).__init__()
        # Use `groups` option to implement depthwise convolution
        # [M, H, K] -> [M, H, K]
        self.skip = skip
        depthwise_conv = nn.Conv1d(in_channels,
                                   in_channels,
                                   kernel_size,
                                   stride=stride,
                                   padding=(padding if pad else 0),
                                   dilation=dilation,
                                   groups=in_channels,
                                   bias=False)
        prelu = nn.PReLU()
        norm = chose_norm(norm_type, in_channels)
        # [M, H, K] -> [M, B, K]
        # Put together
        self.net = nn.Sequential(depthwise_conv, prelu, norm)
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        if skip:
            self.skip_conv = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        """
        Args:
            x: [M, H, K]
        Returns:
            result: [M, B, K]
        """
        out = self.net(x)
        residual = self.res_conv(out)

        skip = 0
        if self.skip:
            skip = self.skip_conv(out)
        return residual, skip


def chose_norm(norm_type, channel_size):
    """The input of normlization will be (M, C, K), where M is batch size,
       C is channel size and K is sequence length.
    """
    if norm_type == "gLN":
        return GlobalLayerNorm(channel_size)
    elif norm_type == "cLN":
        return ChannelwiseLayerNorm(channel_size)
    elif norm_type == "id":
        return nn.Identity()
    else:  # norm_type == "BN":
        # Given input (M, C, K), nn.BatchNorm1d(C) will accumulate statics
        # along M and K, so this BN usage is right.
        return nn.BatchNorm1d(channel_size)


# TODO: Use nn.LayerNorm to impl cLN to speed up
class ChannelwiseLayerNorm(nn.Module):
    """Channel-wise Layer Normalization (cLN)"""
    def __init__(self, channel_size):
        super(ChannelwiseLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            cLN_y: [M, N, K]
        """
        mean = torch.mean(y, dim=1, keepdim=True)  # [M, 1, K]
        var = torch.var(y, dim=1, keepdim=True, unbiased=False)  # [M, 1, K]
        cLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return cLN_y


class GlobalLayerNorm(nn.Module):
    """Global Layer Normalization (gLN)"""
    def __init__(self, channel_size):
        super(GlobalLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.beta = nn.Parameter(torch.Tensor(1, channel_size, 1))  # [1, N, 1]
        self.reset_parameters()

    def reset_parameters(self):
        self.gamma.data.fill_(1)
        self.beta.data.zero_()

    def forward(self, y):
        """
        Args:
            y: [M, N, K], M is batch size, N is channel size, K is length
        Returns:
            gLN_y: [M, N, K]
        """
        # TODO: in torch 1.0, torch.mean() support dim list
        mean = y.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)  # [M, 1, 1]
        var = (torch.pow(y - mean, 2)).mean(dim=1, keepdim=True).mean(dim=2, keepdim=True)
        gLN_y = self.gamma * (y - mean) / torch.pow(var + EPS, 0.5) + self.beta
        return gLN_y


class Spectrogram(nn.Module):
    """
    Calculate the mel spectrogram as an additional input for the encoder
    """
    def __init__(self, n_fft, hop, mels, audio_channels, sr):
        """
        Arguments:
            n_fft {int} -- The number fo frequency bins
            hop {int} -- Hop size (stride)
            mels {int} -- The number of mel filters
            sr {int} -- Sampling rate of the signal
        """
        super(Spectrogram, self).__init__()

        self.n_fft = n_fft
        self.hop = hop
        self.mels = mels
        self.sr = sr
        self.audio_channels = audio_channels

        # Hann window for STFT
        self.window = nn.Parameter(torch.hann_window(n_fft), requires_grad=False)

        # learnable mel transform
        stft_size = n_fft // 2 + 1
        self.mel_transform = nn.Conv1d(2*stft_size,
                                       mels,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=True)

    def forward(self, audio_signal, target_length=None):
        """
        Arguments:
            audio_signal {torch.tensor} -- input tensor of shape (M, audio_channels, T)
        Keyword Arguments:
            target_length {int, None} -- Optional argument for interpolating the time dimension of the result to $target_length (default: {None})
        Returns:
            torch.tensor -- mel spectrogram of shape (M, mels, T')
        """
        mag = self.calculate_mag(audio_signal, db_conversion=True)  # shape: (M, N', T')
        mag = self.mel_transform(mag)  # shape: (M, N, T')

        if target_length is not None:
            mag = F.interpolate(mag, size=target_length, mode='linear',
                                align_corners=True)  # shape: (M, N, T'')

        return mag  # shape: (M, N, T'')

    def calculate_mag(self, input, db_conversion=True):
        """
        Calculate the dB magnitude of the STFT of the input signal
        Arguments:
            audio_signal {torch.tensor} -- input tensor of shape (M, 1, T)
        Keyword Arguments:
            db_conversion {bool} -- True if the method should logaritmically transform the result to dB (default: {True})
        Returns:
            torch.tensor -- output tensor of shape (M, N', T')
        """
        signal = input.view(-1, input.shape[-1])  # shape (M, T)

        stft = torch.stft(signal,
                          n_fft=self.n_fft,
                          hop_length=self.hop,
                          window=self.window,
                          center=True,
                          normalized=False,
                          return_complex=False,
                          onesided=True,
                          pad_mode='reflect')  # shape: (M, N', T', 2)

        mag = (stft**2).sum(-1)  # shape: (M, N', T')
        if db_conversion:
            mag = torch.log10(mag + 1e-8)  # shape: (M, N', T')generate_mpgtf(sample_rate, kernel_size/sample_rate, out_channels, self.center_freqs, self.phase_shifts, True)

        return mag.view(input.shape[0], input.shape[1]*mag.shape[1], -1)  # shape: (M, N', T')


class StrongerEncoder(nn.Module):
    """
    Encodes the waveforms into the latent representation
    """
    def __init__(self, kernel_size, enc_dim, layers, num_mels, audio_channels=2, samplerate=22050, concat_features=True):
        """
        Arguments:
            N {int} -- Dimension of the output latent representation
            kernel_size {int} -- Base convolutional kernel size
            layers {int} -- Number of parallel convolutions with different kernel sizes
            num_mels {int} -- Number of mel filters in the mel spectrogram
            samplerate {int} -- Sampling rate of the input
            cocnat_featuers {bool} -- Determines if we want to merge spectrogram and time domain features
        """
        super(StrongerEncoder, self).__init__()
        self.concat_features = concat_features
        self.spectrogram = Spectrogram(n_fft=1024, hop=256, mels=num_mels, audio_channels=audio_channels, sr=samplerate)

        self.filters = nn.ModuleList([])
        filter_width = num_mels
        for l in range(layers):
            n = enc_dim // 4
            k = kernel_size * (2**l)
            self.filters.append(
                nn.Conv1d(audio_channels,
                          n,
                          kernel_size=k,
                          stride=kernel_size // 2,
                          bias=False,
                          padding=(k - (kernel_size // 2)) // 2))
            filter_width += n

        self.nonlinearity = nn.ReLU()
        if concat_features:
            self.bottleneck = nn.Sequential(
                nn.Conv1d(filter_width, enc_dim, kernel_size=1, stride=1, bias=False),
                nn.ReLU(),
                nn.Conv1d(enc_dim, enc_dim, kernel_size=1, stride=1, bias=False),
            )
        else:
            assert enc_dim % 2 == 0, 'enc_dim should be divisible by two'
            self.timefreq_bottleneck = nn.Sequential(nn.Conv1d(num_mels, enc_dim // 2, kernel_size=1, stride=1, bias=False), nn.ReLU())
            self.time_bottleneck = nn.Sequential(nn.Conv1d(filter_width-num_mels, enc_dim // 2, kernel_size=1, stride=1, bias=False), nn.ReLU())

    def forward(self, signal):
        """
        Arguments:
            signal {torch.tensor} -- mixed signal of shape (M, 1, T)
        Returns:
            torch.tensor -- latent representation of shape (M, N, T)
        """
        convoluted_x = []
        for filter in self.filters:
            x = filter(signal).unsqueeze(-2)  # shape: (M, N^, 1, T')
            convoluted_x.append(x)

        x = torch.cat(convoluted_x, dim=-2)  # shape: (M, N^, L, T')
        x = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3])  # shape: (M, N', T')
        x = self.nonlinearity(x)  # shape: (M, N', T')

        spectrogram = self.spectrogram(signal, x.shape[-1])  # shape: (M, mel, T')
        if self.concat_features:
            x = torch.cat([x, spectrogram], dim=1)  # shape: (M, N*, T')
            return self.bottleneck(x)  # shape: (M, N, T)
        else:
            spectrogram = self.timefreq_bottleneck(spectrogram)
            x = self.time_bottleneck(x)
            x = torch.cat([x, spectrogram], dim=1)  # shape: (M, N, T')
            return x

class StrongerDecoder(nn.Module):
    """
    Decodes the latent representation back to waveforms
    """
    def __init__(self, kernel_size, enc_dim, layers, instr_num=2, audio_channels=2):
        """
        Arguments:
            N {int} -- Dimension of the input latent representation
            kernel_size {int} -- Base convolutional kernel size
            stride {int} -- Stride of the transposed covolutions
            layers {int} -- Number of parallel convolutions with different kernel sizes
        """
        super(StrongerDecoder, self).__init__()

        self.filter_widths = [enc_dim // (2**(l + 1)) for l in range(layers)]
        self.audio_channels = audio_channels
        total_input_width = np.array(self.filter_widths).sum()

        self.bottleneck = nn.Sequential(
            nn.ConvTranspose1d(enc_dim, total_input_width, kernel_size=1, stride=1, bias=False),
            nn.ReLU())
        self.filters = nn.ModuleList([])
        for l in range(layers):
            n = enc_dim // (2**(l + 1))
            k = kernel_size * (2**l)
            self.filters.append(
                nn.ConvTranspose1d(n,
                                   audio_channels,
                                   kernel_size=k,
                                   stride=kernel_size // 2,
                                   bias=False,
                                   padding=(k - (kernel_size // 2)) // 2))

    def forward(self, input):
        """
        Arguments:
            x {torch.tensor} -- Latent representation of the four instrument with shape (M, instr_num, N, T')
        Returns:
            torch.tensor -- Signal of the four instruments with shape (M, instr_num, audio_channels, T)
        """
        x = self.bottleneck(input.view(-1, input.shape[-2], input.shape[-1]))  # shape: (M, instr_num, N', T')

        output = 0.0
        x = x.split(self.filter_widths, dim=-2)
        for i in range(len(x)):
            output += self.filters[i](x[i])  # shape: (M, instr_num, audio_channels, T)
        return output.view(input.shape[0], input.shape[1], self.audio_channels, -1)  # shape: (M, instr_num, audio_channels, T)

class GammatoneEncoder(nn.Module):
    def __init__(self, out_channels, kernel_size, stride, sample_rate=16000, in_channels=2, learnable=False, phase_split=False):
        super(GammatoneEncoder,self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels

        self.sample_rate = sample_rate
        self.learnable=learnable
        
        if learnable:
            self.center_freqs = nn.ParameterList()
            self.phase_shifts = nn.ParameterList()
            generate_mpgtf(sample_rate, kernel_size/sample_rate, out_channels, self.center_freqs, self.phase_shifts, True)
        
        if not learnable:
            self.filters = generate_mpgtf(sample_rate, kernel_size/sample_rate, out_channels, 48, phase_split)
            self.filters = torch.from_numpy(self.filters).cuda().float().unsqueeze(1).expand(-1, in_channels, -1)
        
    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """
        if self.learnable:
            self.filters = generate_mpgtf(self.sample_rate, self.kernel_size/self.sample_rate, self.out_channels, self.center_freqs, self.phase_shifts).cuda().float().unsqueeze(1).expand(-1, self.in_channels, -1)

        filtered_signal = F.conv1d(waveforms, self.filters, stride=self.stride, bias=None, groups=1)
        return F.relu(filtered_signal)
    
class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Forcing the filters to be odd (i.e, perfectly symmetrics)
        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1
            
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.in_channels = in_channels

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        # initialize filterbanks such that they are equally spaced in Mel scale
        low_hz = 30
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)
        

        # filter lower frequency (out_channels, 1)     
#         self.low_hz_ = nn.ParameterList([nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1)) for i in range(in_channels)])
        self.low_hz_ = torch.stack([torch.Tensor(hz[:-1]).view(-1, 1) for i in range(in_channels)], dim=0)
        # filter frequency band (out_channels, 1)
#         self.band_hz_ = nn.ParameterList([nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1)) for i in range(in_channels)])
        self.band_hz_ = torch.stack([torch.Tensor(np.diff(hz)).view(-1, 1) for i in range(in_channels)], dim=0)
        
        # Hamming window
        #self.window_ = torch.hamming_window(self.kernel_size)
        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2))) # computing only half of the window
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size);


        # (1, kernel_size/2)
        n = (self.kernel_size - 1) // 2
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate # Due to symmetry, I only need half of the time axes

                
        torch.manual_seed(5)
        self.phase_offset = torch.rand((out_channels, 1)).repeat(1, self.n_.shape[1])*2*math.pi
        
        
        self.linear = nn.Linear(out_channels, out_channels, bias=False)
        
#         Maybe move to forward
        self.n_ = self.n_.to('cuda')

        self.window_ = self.window_.to('cuda')

        self.low_hz_ = self.low_hz_.to('cuda')
        
        self.band_hz_ = self.band_hz_.to('cuda')
        
        self.phase_offset = self.phase_offset.to('cuda')
        
        
        

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

#         self.n_ = self.n_.to(waveforms.device)

#         self.window_ = self.window_.to(waveforms.device)

        band_pass_filters = []
        for i in range(self.in_channels):
            
            low = self.min_low_hz  + torch.abs(self.low_hz_[i])

            high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_[i]),self.min_low_hz,self.sample_rate/2)
            band=(high-low)[:,0]

            f_times_t_low = torch.matmul(low, self.n_)
            f_times_t_high = torch.matmul(high, self.n_)
            band_pass_left=((torch.sin(f_times_t_high + self.phase_offset)-torch.sin(f_times_t_low + self.phase_offset))/(self.n_/2))*self.window_ # Equivalent of Eq.4 of the reference paper (SPEAKER RECOGNITION FROM RAW WAVEFORM WITH SINCNET). I just have expanded the sinc and simplified the terms. This way I avoid several useless computations. 
            band_pass_center = 2*band.view(-1,1)
            band_pass_right= torch.flip(band_pass_left,dims=[1])


            band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1)


            band_pass = band_pass / (2*band[:,None])
            band_pass_filters.append(band_pass)

        
        self.filters = (torch.stack(band_pass_filters, dim=1)).view(
            self.out_channels, self.in_channels, self.kernel_size)

        filtered_signal = F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)
        return F.relu(self.linear(filtered_signal.transpose(2,1)).transpose(2,1))
    
    
    
if __name__ == "__main__":
    print('I do nothing')
