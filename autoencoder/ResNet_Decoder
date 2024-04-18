import torch
import torch.nn as nn
import torch.nn.functional as F



class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, upsample=None, stride=1, output_padding=0):
        super(DecoderBlock, self).__init__()

        # Upsampling followed by a ConvTranspose1d layer
        self.conv1 = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=3, stride=stride,
                                        padding=1, output_padding=output_padding, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.ConvTranspose1d(out_channels, out_channels, kernel_size=3, stride=1,
                                        padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm1d(out_channels)

        self.upsample = upsample
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.relu(self.batch_norm2(self.conv2(x)))

        if self.upsample is not None:
            x = self.upsample(x)

        return x

class DecoderResNet(nn.Module):
    def __init__(self, num_channels=3, output_dims=1024):
        super(DecoderResNet, self).__init__()

        self.fc = nn.Linear(1024, 512 * Bottleneck.expansion)  # Inverse of encoder's final layer

        # Starting from the deepest layer of encoder and building up
        self.layer4 = self._make_layer(512, 256, stride=2, upsample=True)
        self.layer3 = self._make_layer(256, 128, stride=2, upsample=True)
        self.layer2 = self._make_layer(128, 64, stride=2, upsample=True)
        self.layer1 = self._make_layer(64, 64, stride=2, upsample=True, output_padding=1)  # Adjust padding if necessary

        self.conv1 = nn.ConvTranspose1d(64, num_channels, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.batch_norm1 = nn.BatchNorm1d(num_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 512 * Bottleneck.expansion, 1)  # Reshape to match the spatial dimensions at the encoder's deepest layer

        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)

        x = self.relu(self.batch_norm1(self.conv1(x)))
        return x

    def _make_layer(self, in_channels, out_channels, stride=1, upsample=False, output_padding=0):
        layers = []
        layers.append(DecoderBlock(in_channels, out_channels, upsample=nn.Upsample(scale_factor=stride),
                                   stride=stride, output_padding=output_padding))
        return nn.Sequential(*layers)

# Example usage
decoder = DecoderResNet(num_channels=3, output_dims=1024)
