import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNNet(nn.Module):
    def __init__(self, num_classes):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 9, 3)
        self.conv2 = nn.Conv2d(9, 3, 3)
        self.dropout = nn.Dropout2d(0.25)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(8748, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.fc(x)
        output = torch.sigmoid(x)
        return output


if __name__ == "__main__":
    model = CNNNet(num_classes=1)
    x = model(torch.rand(1, 3, 224, 224))
    print(x)
    print(model)
