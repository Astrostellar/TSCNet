import torch.nn as nn
import torch

class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)

class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])
        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning

class RDN(nn.Module):
    def __init__(self, in_dim=1, feature_dim=128, num_features=64, growth_rate=64, num_blocks=8, num_layers=3):
        super(RDN, self).__init__()
        self.G0 = num_features
        self.G = growth_rate
        self.D = num_blocks
        self.C = num_layers
        # shallow feature extraction
        self.sfe1 = nn.Conv2d(in_dim, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)
        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))
        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        self.output = nn.Conv2d(self.G0, feature_dim, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)
        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)
        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.output(x)
        return x
    

class InterpolationNetwork(nn.Module):
    def __init__(self):
        super(InterpolationNetwork, self).__init__()
        self.feature_extract_1 = RDN(feature_dim=64, num_blocks=3, num_layers=5)
        self.feature_extract_2 = RDN(feature_dim=64, num_blocks=3, num_layers=5)
        self.slice_reconstruction = RDN(in_dim=128, feature_dim=1, num_blocks=3, num_layers=5)

    def forward(self, x1, x2):
        feature_1 = self.feature_extract_1(x1)
        feature_2 = self.feature_extract_2(x2)
        feature = torch.cat([feature_1, feature_2], 1)
        out = self.slice_reconstruction(feature)
        return out


class DiscriminatorForVGG(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            channels: int,
    ) -> None:
        super(DiscriminatorForVGG, self).__init__()
        self.features = nn.Sequential(
            # input size. (3) x 96 x 96
            nn.Conv2d(in_channels, channels, (3, 3), (1, 1), (1, 1), bias=True),
            nn.LeakyReLU(0.2, True),
            # state size. (64) x 48 x 48
            nn.Conv2d(channels, channels, (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(channels, int(2 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(2 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (128) x 24 x 24
            nn.Conv2d(int(2 * channels), int(2 * channels), (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(2 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(2 * channels), int(4 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(4 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (256) x 12 x 12
            nn.Conv2d(int(4 * channels), int(4 * channels), (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(4 * channels)),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(int(4 * channels), int(8 * channels), (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
            # state size. (512) x 6 x 6
            nn.Conv2d(int(8 * channels), int(8 * channels), (3, 3), (2, 2), (1, 1), bias=False),
            nn.BatchNorm2d(int(8 * channels)),
            nn.LeakyReLU(0.2, True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(int(8 * channels) * 16 * 16, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, out_channels),
        )

    def forward(self, x):
        # Input image size must equal 96
        # assert x.size(2) == 96 and x.size(3) == 96, "Input image size must be is 96x96"

        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x
