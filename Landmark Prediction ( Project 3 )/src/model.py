import torch
import torch.nn as nn
import torch.nn.functional as F


# define the CNN architecture

class MyModel(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(MyModel, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512, 1024, 3, padding=1)
        self.bn8 = nn.BatchNorm2d(1024)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # Global average pooling with output size 4x4
        self.fc1 = nn.Linear(1024, num_classes)  # Adjusted input size after global average pooling
#         self.bn6 = nn.BatchNorm1d(512)
#         self.fc2 = nn.Linear(256, 128)
#         self.bn7 = nn.BatchNorm1d(128)
#         self.fc3 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.dropout2d = nn.Dropout2d(dropout)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.pool1(self.leakyrelu(self.bn1(self.conv1(x))))
        x = self.pool1(self.leakyrelu(self.bn2(self.conv2(x))))
        x = self.dropout2d(x)
        x = self.pool1(self.leakyrelu(self.bn3(self.conv3(x))))
        x = self.pool2(self.leakyrelu(self.bn4(self.conv4(x))))
        x = self.dropout2d(x)
        x = self.pool2(self.leakyrelu(self.bn8(self.conv5(x))))
        x = self.dropout2d(x)
        x = self.avgpool(x)  # Apply global average pooling with output size 4x4
        x = x.view(x.size(0), -1)  # Flatten the tensor

        x = self.fc1(x)
#         x = self.dropout(x)
#         x = self.leakyrelu(self.bn7(self.fc2(x)))
#         x = self.dropout(x)
#         x = self.fc3(x)

        return x

    
    
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
