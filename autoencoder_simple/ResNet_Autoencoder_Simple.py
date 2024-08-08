import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder_Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Encoder_Bottleneck, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm1d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)
        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, layer_list, num_channels=9):
        super(Encoder, self).__init__()
        self.in_channels = num_channels
        
        self.conv0 = nn.Conv1d(self.in_channels, 32, kernel_size=1, stride=1, padding=0)
        self.layer1 = self._make_encoder_layer(layer_list[0], planes=64)
        self.conv2 = nn.Conv1d(self.in_channels, 14, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        x = self.conv0(x)
        x = self.layer1(x)
        x = self.conv2(x)
        print(x.shape)
        return x
    
    def _make_encoder_layer(self, blocks, planes, stride=1, downsample_padding = 0):
        ii_downsample = None
        layers = []
        
        if stride == 1 and self.in_channels != planes*Encoder_Bottleneck.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, planes*Encoder_Bottleneck.expansion, kernel_size=1, padding =0, stride=stride),
                nn.BatchNorm1d(planes*Encoder_Bottleneck.expansion)
            )
        elif stride != 1:
            ii_downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, planes*Encoder_Bottleneck.expansion, kernel_size=1, padding = downsample_padding, stride=stride),
                nn.BatchNorm1d(planes*Encoder_Bottleneck.expansion)
            )
            
        layers.append(Encoder_Bottleneck(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*Encoder_Bottleneck.expansion
        
        for i in range(blocks-1):
            layers.append(Encoder_Bottleneck(self.in_channels, planes))
            
        return nn.Sequential(*layers)
    
    
class Decoder_Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_upsample=None, stride=1, output_padding=0, last_layer_of_block = False):
        super(Decoder_Bottleneck, self).__init__()

        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, output_padding=output_padding)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.ConvTranspose1d(out_channels, out_channels * (self.expansion - 2 * last_layer_of_block), kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm1d(out_channels * (self.expansion - 2 * last_layer_of_block))

        self.i_upsample = i_upsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))
        
        if self.i_upsample is not None:
            identity = self.i_upsample(identity)

        x += identity
        x = self.relu(x)
        print(x.shape)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layer_list, num_channels=9):
        super(Decoder, self).__init__()
        self.in_channels_decode = 14

        # self.layer5 = self._make_decoder_layer(layer_list[3], planes=512, stride=2)
        # self.layer6 = self._make_decoder_layer(layer_list[2], planes=256, stride=2, output_padding=1)
        # self.layer7 = self._make_decoder_layer(layer_list[1], planes=128, stride=2, output_padding=1)
        self.conv9 = nn.Conv1d(14, 128, kernel_size=1, stride=1, padding=0)
        self.layer8 = self._make_decoder_layer(layer_list[0], planes=64, stride=2)
        self.conv9 = nn.Conv1d(32, 14, kernel_size=1, stride=1, padding=0)
        # self.lastblock = nn.Sequential(
        #     nn.ConvTranspose1d(self.in_channels_decode, 64, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(64, 64, kernel_size=3, stride=1, padding=1, output_padding=0),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(64, num_channels, kernel_size=1, stride=1, padding=0),
        # )

    def forward(self, x):
        # x = self.layer5(x)
        # x = self.layer6(x)
        # x = self.layer7(x)
        x = self.layer8(x)

        # x = self.lastblock(x)
        return nn.Sigmoid(x)
    
    def _make_decoder_layer(self, blocks, planes, stride=1, output_padding = 0, last_layer = False, upsample_padding = 0, upsample_output_padding = 0):
        ii_upsample = None
        layers = []
        
        for i in range(blocks-1):
            layers.append(Decoder_Bottleneck(self.in_channels_decode, planes))
        
        if last_layer:
            return nn.Sequential(*layers)
        
        if stride != 1 or self.in_channels_decode != planes*Decoder_Bottleneck.expansion:
            ii_upsample = nn.Sequential(
                nn.Upsample(scale_factor=2),
                nn.ConvTranspose1d(self.in_channels_decode, planes*(Decoder_Bottleneck.expansion - 2), kernel_size=1, stride=1, padding=upsample_padding, output_padding=upsample_output_padding),
                nn.BatchNorm1d(planes*(Decoder_Bottleneck.expansion-2))
            )

        layers.append(Decoder_Bottleneck(self.in_channels_decode, planes, i_upsample=ii_upsample, stride=stride, output_padding=output_padding, last_layer_of_block=True))
        # print('decoder',self.in_channels_decode)
        self.in_channels_decode = planes*(Decoder_Bottleneck.expansion - 2)
        # print('decoder',self.in_channels_decode)

        return nn.Sequential(*layers)




class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder([1],14)
        self.decoder = Decoder([1],14)

         
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x