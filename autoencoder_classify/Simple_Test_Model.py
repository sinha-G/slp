import torch
import torch.nn as  nn
import torch.nn.functional as F

# https://github.com/JayPatwardhan/ResNet-PyTorch


        
# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv1d(15,90, kernel_size=4, stride=4, padding=0, bias=False)
#         # self.conv2 = nn.LazyConv1d( 90, kernel_size=3, stride = 2)
#         self.conv3 = nn.Conv1d(90, 90, kernel_size=3, stride = 2)
#         self.conv4 = nn.Conv1d(90, 90, kernel_size=3, stride = 2)
#         self.conv5 = nn.Conv1d(90, 20, kernel_size=3, stride = 1)
#         self.fc1 = nn.LazyLinear(128)  # Adjusted based on output shape from conv and pool layers
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 5)

#     def forward(self, x):
#         x = F.relu(self.conv1(x))
#         # x = F.relu(self.conv2(x))
#         x = F.relu(self.conv3(x))
#         x = F.relu(self.conv4(x))
#         x = F.relu(self.conv5(x))
#         # x = self.pool1(x)
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         # x = F.sigmoid(self.fc3(x))
#         x = self.fc3(x)  # Output raw logits
        
#         return x

        
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.LazyConv1d(15,90, kernel_size=4, stride=4, padding=0, bias=False)
        # self.conv2 = nn.LazyConv1d( 90, kernel_size=3, stride = 2)
        # self.conv3 = nn.Conv1d(90, 90, kernel_size=3, stride = 2)
        # self.conv4 = nn.Conv1d(90, 90, kernel_size=3, stride = 2)
        # self.conv5 = nn.Conv1d(90, 20, kernel_size=3, stride = 1)
        self.fc1 = nn.LazyLinear(128)  # Adjusted based on output shape from conv and pool layers
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.relu(self.conv2(x))
        # x = F.relu(self.conv3(x))
        # x = F.relu(self.conv4(x))
        # x = F.relu(self.conv5(x))
        # x = self.pool1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.sigmoid(self.fc3(x))
        x = self.fc3(x)  # Output raw logits
        
        return x
