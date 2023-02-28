import torch

from torch import nn


class Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Teacher computational complexity: 18.91 MMac
        Teacher number of parameters: .36 k
        """
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(8, 16, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 16, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 16, 3, stride=2, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 8, 3, stride=2, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Flatten(start_dim=1),
            nn.Linear(128, 20),
        )

        self.decoder = nn.Sequential(
            nn.Linear(20, 8 * 4 * 4),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Unflatten(dim=1, unflattened_size=(8, 4, 4)),
            nn.ConvTranspose2d(
                8, 16, 3, stride=2, padding=1, output_padding=1
            ),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(
                16, 16, 3, stride=2, padding=1, output_padding=1
            ),
            nn.LeakyReLU(negative_slope=0.3),
            nn.ConvTranspose2d(
                16, 16, 3, stride=2, padding=1, output_padding=1
            ),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 8, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(8, 1, 3, stride=1, padding="valid"),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.sigmoid(x)
        return x


class StudentA(nn.Module):
    """
    Student computational complexity: 6.37 MMac
    Student number of parameters: 7.18 k
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(8, 16, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 16, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 16, 3, stride=2, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(16, 8, 3, stride=2, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Flatten(start_dim=1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class StudentB(nn.Module):
    """
    Student computational complexity: 2.25 MMac
    Student number of parameters: 2.19 k
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(8, 8, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(8, 8, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(8, 8, 3, stride=2, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(8, 4, 3, stride=2, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Flatten(start_dim=1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class StudentC(nn.Module):
    """
    Student computational complexity: 432.09 KMac
    Student number of parameters: 1.06 k
    """

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.AvgPool2d(2),
            nn.Conv2d(8, 8, 3, stride=1, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Conv2d(8, 4, 3, stride=2, padding="valid"),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Flatten(start_dim=1),
            nn.Linear(100, 1),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x


class StudentD(nn.Module):
    """
    Student computational complexity: 131.26 KMac
    Student number of parameters: 409
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, stride=1, padding="valid")
        self.relu1 = nn.LeakyReLU(negative_slope=0.3)
        self.maxp1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(4, 8, 3, stride=2, padding="valid")
        self.relu2 = nn.LeakyReLU(negative_slope=0.3)
        self.maxp2 = nn.AvgPool2d(2)
        self.linear = nn.Linear(72, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxp2(x)
        x = nn.Flatten(start_dim=1)(x)
        x = self.linear(x)
        return x


class StudentE(nn.Module):
    """
    Student computational complexity: 114.3 KMac
    Student number of parameters: 225
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, stride=1, padding="valid")
        self.relu1 = nn.LeakyReLU(negative_slope=0.3)
        self.maxp1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(4, 4, 3, stride=2, padding="valid")
        self.relu2 = nn.LeakyReLU(negative_slope=0.3)
        self.maxp2 = nn.AvgPool2d(2)
        self.linear = nn.Linear(36, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxp2(x)
        x = nn.Flatten(start_dim=1)(x)
        x = self.linear(x)
        return x


class StudentF(nn.Module):
    """
    Student computational complexity: 57.85 KMac
    Student number of parameters: 133
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, stride=1, padding="valid")
        self.relu1 = nn.LeakyReLU(negative_slope=0.3)
        self.maxp1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(2, 4, 3, stride=2, padding="valid")
        self.relu2 = nn.LeakyReLU(negative_slope=0.3)
        self.maxp2 = nn.AvgPool2d(2)
        self.linear = nn.Linear(36, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxp2(x)
        x = nn.Flatten(start_dim=1)(x)
        x = self.linear(x)
        return x


class StudentG(nn.Module):
    """
    Student computational complexity: 53.27 KMac
    Student number of parameters: 77
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, stride=1, padding="valid")
        self.relu1 = nn.LeakyReLU(negative_slope=0.3)
        self.maxp1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(2, 2, 3, stride=2, padding="valid")
        self.relu2 = nn.LeakyReLU(negative_slope=0.3)
        self.maxp2 = nn.AvgPool2d(2)
        self.linear = nn.Linear(18, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxp1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxp2(x)
        x = nn.Flatten(start_dim=1)(x)
        x = self.linear(x)
        return x
