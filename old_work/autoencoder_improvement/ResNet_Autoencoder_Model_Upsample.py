import torch
import torch.nn as  nn
import torch.nn.functional as F

# https://github.com/JayPatwardhan/ResNet-PyTorch

class Encoder_Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1, lengths=[]):
        super(Encoder_Bottleneck, self).__init__()
        self.lengths = lengths
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)

        # self.pool2 = nn.MaxPool1d(kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm1d(out_channels * self.expansion)

        self.i_downsample = i_downsample
        self.ELU = nn.ELU()

    def forward(self, x):
        identity = x.clone()
        x = self.ELU(self.batch_norm1(self.conv1(x)))
        x = self.ELU(self.batch_norm2(self.conv2(x)))
        x = self.conv3(x)
        x = self.batch_norm3(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)
            # print('identity',identity)

        x += identity
        self.lengths.append(x.shape[-1])
        # print(x.shape[-1])
        # print('identity',identity.shape)
        # print('x',x.shape)
        x = self.ELU(x)
        return x
    
class Decoder_Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, i_upsample=None, stride=1, output_padding=0, last_layer_of_block = False,lengths=[]):
        super(Decoder_Bottleneck, self).__init__()
        
        self.lengths = lengths
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)

        self.upsample1 = nn.Upsample(scale_factor=stride)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels * (self.expansion - 2 * last_layer_of_block), kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm1d(out_channels * (self.expansion - 2 * last_layer_of_block))

        self.i_upsample = i_upsample
        self.ELU = nn.ELU()

    def forward(self, x):
        identity = x.clone()
        size = self.lengths.pop()
        # print(size)
        x = self.ELU(self.batch_norm1(self.conv1(x)))
        x = self.upsample1(x)
        
        if x.shape[-1] == size + 1:
            x = x[:,:,:-1]
        
        x = self.ELU(self.batch_norm2(self.conv2(x)))
        x = self.batch_norm3(self.conv3(x))
        if self.i_upsample is not None:
            identity = self.i_upsample(identity)
            if identity.shape[-1] == size + 1:
                identity = identity[:,:,:-1]
                # print(identity.shape)
        
        # print('x',x)
            # print('identity',identity)

        x += identity
        
        # print('pop',len(self.residuals))
        
        # print('identity',identity.shape)
        # print('x', x.shape)
        # x += identity
        
        x = self.ELU(x)
        return x

        
class ResNet(nn.Module):
    def __init__(self, layer_list, num_channels=9):
        super(ResNet, self).__init__()
        
        #################### Encoder Part ####################
        self.in_channels = num_channels
        self.lengths =  []
        # Think about putting some layers before the blocks:
        # self.begin = nn.Sequential(
        #     nn.Conv1d(9, 64, 1),
        #     nn.BatchNorm1d(64),
        #     nn.ELU()
        # )
                
        # Blocks
        self.layer1 = self._make_encoder_layer(layer_list[0], planes=64)
        self.layer2 = self._make_encoder_layer(layer_list[1], planes=128, stride=2)
        self.layer3 = self._make_encoder_layer(layer_list[2], planes=256, stride=2)
        self.layer4 = self._make_encoder_layer(layer_list[3], planes=512, stride=2)
        
      
        # What shape do we want it to have after encoding?
        # How do we turn it into the shape we want after the last block?
        # Idea: Linear layers?
        self.reduce = nn.Sequential(
            nn.Linear(512 * Encoder_Bottleneck.expansion * 8, 64),
            # nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Linear(64,64),
            # nn.BatchNorm1d(64),
            # nn.ELU()
        )
        
        ###################### Decoder Part ######################
        self.expand = nn.Sequential(
            nn.Linear(64,64),
            # nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Linear(64,512 * Encoder_Bottleneck.expansion * 8),
            # nn.BatchNorm1d(512 * Encoder_Bottleneck.expansion * 8),
            nn.ELU()
        )

        self.layer5 = self._make_decoder_layer(layer_list[3], planes=512, stride=2)
        self.layer6 = self._make_decoder_layer(layer_list[2], planes=256, stride=2, output_padding=1)
        self.layer7 = self._make_decoder_layer(layer_list[1], planes=128, stride=2, output_padding=1)
        self.layer8 = self._make_decoder_layer(layer_list[0], planes=64, last_layer=True)
        
        self.lastblock = nn.Sequential(
            nn.Conv1d(self.in_channels, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Conv1d(64, num_channels, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        ###################### Encoder Part ######################
        # print(self.in_channels)
        # x = self.begin(x)
        x = self.layer1(x)
        # print(self.in_channels)
        x = self.layer2(x)
        # print(self.in_channels)
        x = self.layer3(x)
        # print(self.in_channels)
        x = self.layer4(x)
        # x = self.layer5(x)
        # x = self.layer6(x)
        # print(self.in_channels)
        x = x.reshape(x.size(0), -1,)

        f = nn.Identity()
        x = f(x)

        x = self.reduce(x)
        #print(x.shape)
        
        ####################### Decoder Part #######################
        # print()
        # for y in self.residuals:
        #     print(y.shape)
        # print()
        x = self.expand(x)
        # print(x.shape)
        # print(self.in_channels)
        
        x = torch.reshape(x, (x.size(0), 2048, 8))
        # x = x.view(512 * Encoder_Bottleneck.expansion,8)
        # print(x.shape)
        self.lengths.pop()
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        
        x = self.lastblock(x)

        return x
        
    def _make_encoder_layer(self, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*Encoder_Bottleneck.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, planes*Encoder_Bottleneck.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm1d(planes*Encoder_Bottleneck.expansion)
            )
            
        layers.append(Encoder_Bottleneck(self.in_channels, planes, i_downsample=ii_downsample, stride=stride,lengths=self.lengths))
        self.in_channels = planes*Encoder_Bottleneck.expansion
        
        for i in range(blocks-1):
            layers.append(Encoder_Bottleneck(self.in_channels, planes, lengths=self.lengths))
            
        return nn.Sequential(*layers)
    
    def _make_decoder_layer(self, blocks, planes, stride=1, output_padding = 0, last_layer = False):
        ii_upsample = None
        layers = []
        
        for i in range(blocks-1):
            layers.append(Decoder_Bottleneck(self.in_channels, planes,lengths=self.lengths))
        
        if last_layer:
            return nn.Sequential(*layers)
        
        if stride != 1 or self.in_channels != planes*Decoder_Bottleneck.expansion:
            ii_upsample = nn.Sequential(
                # nn.Conv1d(self.in_channels, planes*(Decoder_Bottleneck.expansion - 2), kernel_size=1, stride=stride, output_padding=output_padding),
                nn.Upsample(scale_factor=stride),
                nn.Conv1d(self.in_channels, planes*(Decoder_Bottleneck.expansion - 2), kernel_size=1, stride=1),
                nn.BatchNorm1d(planes*(Decoder_Bottleneck.expansion-2))
            )

        layers.append(Decoder_Bottleneck(self.in_channels, planes, i_upsample=ii_upsample, stride=stride, output_padding=output_padding, last_layer_of_block=True, lengths=self.lengths))
        self.in_channels = planes*(Decoder_Bottleneck.expansion - 2)

        return nn.Sequential(*layers)

        
        
def ResNet_Autoencoder( channels=9):
    # return ResNet([3,4,6,3],  channels)
    # return ResNet([2,3,4,2],  channels)
    return ResNet([2,2,2,2],  channels)
    


