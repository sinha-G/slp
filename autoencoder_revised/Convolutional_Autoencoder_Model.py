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
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class Encoder(nn.Module):
    def __init__(self, layer_list, num_channels=9, bottleneck_size=60):
        super(Encoder, self).__init__()
        self.in_channels = num_channels
        self.bottleneck_size = bottleneck_size
        
        self.layer1 = self._make_encoder_layer(layer_list[0], planes=64)
        self.layer2 = self._make_encoder_layer(layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_encoder_layer(layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_encoder_layer(layer_list[3], planes=512, stride=2)
        
        self.reduce5 = nn.Sequential(
            nn.Conv1d(4*512, self.bottleneck_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.linear6 = nn.Sequential(
            nn.Linear(in_features=3375, out_features=2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048)
        )
        # self.linear6 = nn.Sequential(
        #     nn.Linear(in_features=3375, out_features=128),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(128)
        # )
        # self.linear6 = nn.Sequential(
        #     nn.Linear(in_features=3375, out_features=128),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(128)
        # )
        
    def forward(self, x):
        x = self.layer1(x)
        # print('layer1', x.shape)
        x = self.layer2(x)
        # print('layer2', x.shape)
        x = self.layer3(x)
        # print('layer3', x.shape)
        x = self.layer4(x)
        # print('layer4', x.shape)
        x = self.reduce5(x)
        # print('reduce5', x.shape)
        x = x.reshape(x.shape[0], -1)
        # print('reshape', x.shape)
        x = self.linear6(x)
        # print('linear6', x.shape)
        return x
    
    def _make_encoder_layer(self, blocks, planes, stride=1, downsample_padding = 0):
        ii_downsample = None
        layers = []
        
        if stride == 1 and self.in_channels != planes*Encoder_Bottleneck.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, planes*Encoder_Bottleneck.expansion, kernel_size=1, padding=0, stride=stride),
                nn.BatchNorm1d(planes*Encoder_Bottleneck.expansion)
            )
        elif stride != 1:
            ii_downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, planes*Encoder_Bottleneck.expansion, kernel_size=1, padding = downsample_padding, stride=stride),
                nn.BatchNorm1d(planes*Encoder_Bottleneck.expansion)
            )
            
        layers.append(Encoder_Bottleneck(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*Encoder_Bottleneck.expansion
        
        for _ in range(blocks-1):
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
        return x
    
class Decoder(nn.Module):
    def __init__(self, layer_list, num_channels=9, bottleneck_size=60):
        super(Decoder, self).__init__()
        self.in_channels_decode = 2048
        self.bottleneck_size = bottleneck_size
        # self.identity_length = [60,30,15]
        self.identity_length = [3600, 1800, 900]
        
        self.linear = nn.Sequential(
            nn.Linear(2048, 3375),
            nn.ReLU(),
            nn.BatchNorm1d(3375)
        )
        # self.linear = nn.Sequential(
        #     nn.Linear(128, 3375),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(3375)
        # )
        # self.linear = nn.Sequential(
        #     nn.Linear(64, 3375),
        #     nn.ReLU(),
        #     nn.BatchNorm1d(3375)
        # )
        self.expand = nn.Sequential(
            nn.ConvTranspose1d(self.bottleneck_size, 4*512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm1d(4*512),
            nn.ReLU()
        )
        self.layer5 = self._make_decoder_layer(layer_list[3], planes=512, stride=2,output_padding=1)
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
        # print('decoder input', x.shape)
        x = self.linear(x)
        # print('decoder linear', x.shape)
        x = torch.reshape(x, (x.size(0), 15, 225))
        # print('decoder reshape', x.shape)
        x = self.expand(x)
        # print('decoder expand', x.shape)
        x = self.layer5(x)
        # print('decoder layer5', x.shape)
        x = self.layer6(x)
        # print('decoder layer6', x.shape)
        x = self.layer7(x)
        # print('decoder layer7', x.shape)
        x = self.layer8(x)
        # print('decoder layer8', x.shape)

        x = self.lastblock(x)
        # print('decoder lastblock', x.shape)
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
                nn.Upsample(size=self.identity_length.pop()),
                nn.ConvTranspose1d(self.in_channels_decode, planes*(Decoder_Bottleneck.expansion - 2), kernel_size=1, stride=1, padding=upsample_padding, output_padding=upsample_output_padding),
                nn.BatchNorm1d(planes*(Decoder_Bottleneck.expansion-2))
            )

        layers.append(Decoder_Bottleneck(self.in_channels_decode, planes, i_upsample=ii_upsample, stride=stride, output_padding=output_padding, last_layer_of_block=True))
        self.in_channels_decode = planes*(Decoder_Bottleneck.expansion - 2)

        return nn.Sequential(*layers)
    
           
class Autoencoder(nn.Module):
    def __init__(self, layer_list, num_channels=9):
        super(Autoencoder, self).__init__()
        self.num_channels = num_channels
        self.bottleneck_size = 15
        self.encoder = Encoder(layer_list, self.num_channels, self.bottleneck_size)
        self.decoder = Decoder(layer_list, self.num_channels, self.bottleneck_size)

    def forward(self, x):
        # print(self.in_channels)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

        

    
        
        
def ResNet_Autoencoder(channels=13):
    # return Autoencoder([3,4,23,3],  channels)
    return Autoencoder([3,4,6,3], channels)
    # return Autoencoder([2,2,2,2],  channels)
    


