import torch
import torch.nn as  nn
import torch.nn.functional as F

# https://github.com/JayPatwardhan/ResNet-PyTorch

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
            # print()
            # print('identity.shape', identity.shape)
            # print('x.shape', x.shape)
            # print(x)
            identity = self.i_downsample(identity)
            # print('downsample(identity.shape)', identity.shape)
            # print(identity)
        
        x += identity
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, layer_list, num_channels=9):
        super(Encoder, self).__init__()
        self.in_channels = num_channels

        self.layer1 = self._make_encoder_layer(layer_list[0], planes=64)
        self.layer2 = self._make_encoder_layer(layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_encoder_layer(layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_encoder_layer(layer_list[3], planes=512, stride=2, downsample_padding=0)
        
        self.reduce = nn.Sequential(
            nn.Linear(512 * Encoder_Bottleneck.expansion * 8, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
    def forward(self, x):
        # self.in_channels = 13
        # print(self.in_channels)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = x.reshape(x.size(0), -1,)
        
        x = self.reduce(x)
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
        # print(self.in_channels)
        self.in_channels = planes*Encoder_Bottleneck.expansion
        # print(self.in_channels)
        
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
            print()
            print('identity.shape', identity.shape)
            print('x.shape', x.shape)
            # print(x)
            identity = self.i_upsample(identity)
            print('upsample(identity.shape)', identity.shape)
            # print(identity)

        x += identity
        x = self.relu(x)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layer_list, num_channels=9):
        super(Decoder, self).__init__()
        self.in_channels_decode = 2048
        # self.num_channels = num_channels

        self.expand = nn.Sequential(
            nn.Linear(64,64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,512 * Encoder_Bottleneck.expansion * 8),
            # nn.BatchNorm1d(512 * Encoder_Bottleneck.expansion * 8),
            nn.ReLU()
        )

        self.layer5 = self._make_decoder_layer(layer_list[3], planes=512, stride=2)
        self.layer6 = self._make_decoder_layer(layer_list[2], planes=256, stride=2, output_padding=1)
        self.layer7 = self._make_decoder_layer(layer_list[1], planes=128, stride=2, output_padding=1)
        self.layer8 = self._make_decoder_layer(layer_list[0], planes=64, last_layer=True)
        
        self.lastblock = nn.Sequential(
            nn.ConvTranspose1d(self.in_channels_decode, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.ConvTranspose1d(64, num_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        print()
        print('-------------- Decoder -------------')
        x = self.expand(x)
        
        x = torch.reshape(x, (x.size(0), 2048, 8))
        
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)

        x = self.lastblock(x)
        return x
    
    def _make_decoder_layer(self, blocks, planes, stride=1, output_padding = 0, last_layer = False, upsample_padding = 0, upsample_output_padding = 0):
        ii_upsample = None
        layers = []
        
        for i in range(blocks-1):
            layers.append(Decoder_Bottleneck(self.in_channels_decode, planes))
        
        if last_layer:
            return nn.Sequential(*layers)
        
        if stride != 1 or self.in_channels_decode != planes*Decoder_Bottleneck.expansion:
            ii_upsample = nn.Sequential(
                nn.Upsample(scale_factor = (1,stride)),
                nn.ConvTranspose1d(self.in_channels_decode, planes*(Decoder_Bottleneck.expansion - 2), kernel_size=1, stride=1, padding=upsample_padding, output_padding=upsample_output_padding),
                nn.BatchNorm1d(planes*(Decoder_Bottleneck.expansion-2))
            )

        layers.append(Decoder_Bottleneck(self.in_channels_decode, planes, i_upsample=ii_upsample, stride=stride, output_padding=output_padding, last_layer_of_block=True))
        # print('decoder',self.in_channels_decode)
        self.in_channels_decode = planes*(Decoder_Bottleneck.expansion - 2)
        # print('decoder',self.in_channels_decode)

        return nn.Sequential(*layers)
    
           
class Autoencoder(nn.Module):
    def __init__(self, layer_list, num_channels=9):
        super(Autoencoder, self).__init__()
        self.num_channels = num_channels
        self.encoder = Encoder(layer_list, self.num_channels)
        self.decoder = Decoder(layer_list, self.num_channels)

    def forward(self, x):
        # print(self.in_channels)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

        

    


        
        
def ResNet_Autoencoder( channels=13):
    # return ResNet([3,4,6,3],  channels)
    return Autoencoder([1,1,1,1],  channels)
    


