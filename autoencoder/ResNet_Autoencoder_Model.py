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

        
class ResNet(nn.Module):
    def __init__(self, layer_list, num_channels=9):
        super(ResNet, self).__init__()
        
        #################### Encoder Part ####################
        self.in_channels = num_channels

        # Think about putting some layers before the blocks:
        # self.begin = nn.Sequential(
        #     nn.Conv1d(self.in_channels, 128, 1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU()
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
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,27),
            nn.BatchNorm1d(27),
            nn.ReLU()
        )
        
        ###################### Decoder Part ######################
        self.expand = nn.Sequential(
            nn.Linear(27,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,512 * Encoder_Bottleneck.expansion * 8),
            nn.BatchNorm1d(512 * Encoder_Bottleneck.expansion * 8),
            nn.ReLU()
        )

    def forward(self, x):
        ###################### Encoder Part ######################
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # x = self.layer5(x)
        # x = self.layer6(x)
        
        x = x.reshape(x.size(0),-1)
        x = self.reduce(x)
        
        ####################### Decoder Part #######################
        x = self.expand(x)
        x = x.view(512 * Encoder_Bottleneck.expansion,8)


        return x

        
    def _make_encoder_layer(self, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*Encoder_Bottleneck.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, planes*Encoder_Bottleneck.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm1d(planes*Encoder_Bottleneck.expansion)
            )
            
        layers.append(Encoder_Bottleneck(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*Encoder_Bottleneck.expansion
        
        for i in range(blocks-1):
            layers.append(Encoder_Bottleneck(self.in_channels, planes))
            
        return nn.Sequential(*layers)

        
        
def ResNet_Autoencoder( channels=9):
    return ResNet([3,4,6,3,2,1],  channels)
    


