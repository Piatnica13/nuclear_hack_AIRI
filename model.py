import torch
import torch.nn as nn

from config import IN_CHANNELS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE = 48
BOTTLENECK = 128



class ConvBlock(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                cin,
                cout,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                min(8, cout),
                cout,
            ),

            nn.GELU(),

        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()

        self.block = nn.Sequential(

            ConvBlock(cin, cout),
            ConvBlock(cout, cout),

            nn.Conv2d(
                cout,
                cout,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                min(8, cout),
                cout,
            ),

            nn.GELU(),

        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                cin,
                cout,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                min(8, cout),
                cout,
            ),

            nn.GELU(),

        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):

    def __init__(self, cin, cout):
        super().__init__()

        self.up = nn.Sequential(

            nn.Upsample(
                scale_factor=2,
                mode="bilinear",
                align_corners=False,
            ),

            nn.Conv2d(
                cin,
                cout,
                kernel_size=3,
                padding=1,
                bias=False,
            ),

            nn.GroupNorm(
                min(8, cout),
                cout,
            ),

            nn.GELU(),

        )

        self.refine = nn.Sequential(

            ConvBlock(cout, cout),
            ConvBlock(cout, cout),

        )

    def forward(self, x):

        x = self.up(x)
        x = self.refine(x)

        return x


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()

        self.conv1 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.norm1 = nn.GroupNorm(
            min(8, channels),
            channels,
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )

        self.norm2 = nn.GroupNorm(
            min(8, channels),
            channels,
        )

        self.act = nn.GELU()

    def forward(self, x):

        identity = x

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)

        x = x + identity
        x = self.act(x)

        return x


class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.input_shape = None
        self.latent_shape = None

        self.input_elements = None
        self.latent_elements = None

        self.compression_ratio = None

        self.enc0 = ConvBlock(
            IN_CHANNELS,
            BASE,
        )

        self.enc1 = EncoderBlock(
            BASE,
            BASE * 2,
        )

        self.enc2 = EncoderBlock(
            BASE * 2,
            BASE * 4,
        )

        self.enc3 = EncoderBlock(
            BASE * 4,
            BOTTLENECK,
        )

        self.enc4 = EncoderBlock(
            BOTTLENECK,
            BOTTLENECK,
        )

        self.bottleneck = nn.Sequential(

            ResidualBlock(BOTTLENECK),
            ResidualBlock(BOTTLENECK),
            ResidualBlock(BOTTLENECK),
            ResidualBlock(BOTTLENECK),
            ResidualBlock(BOTTLENECK),

        )

        self.dec4 = Up(
            BOTTLENECK,
            BOTTLENECK,
        )

        self.dec3 = Up(
            BOTTLENECK,
            BASE * 4,
        )

        self.dec2 = Up(
            BASE * 4,
            BASE * 2,
        )

        self.dec1 = Up(
            BASE * 2,
            BASE,
        )

        self.head = nn.Conv2d(
            BASE,
            IN_CHANNELS,
            kernel_size=1,
        )

        self._printed = False

    def forward(self, x):

        identity = x
    
        skip = self.enc0(x)
    
        x = self.enc1(skip)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
    
        latent = self.bottleneck(x)
    
        x = self.dec4(latent)
        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)
    
        x = x + skip
    
        reconstruction = self.head(x)
        reconstruction = reconstruction + identity
    

        if not self._printed:

            self.input_shape = tuple(identity.shape[1:])
            self.latent_shape = tuple(latent.shape[1:])

            self.input_elements = (
                identity.shape[1]
                * identity.shape[2]
                * identity.shape[3]
            )

            self.latent_elements = (
                latent.shape[1]
                * latent.shape[2]
                * latent.shape[3]
            )

            self.compression_ratio = (
                self.input_elements
                / self.latent_elements
            )

            params = sum(
                p.numel()
                for p in self.parameters()
                if p.requires_grad
            )

            print("=" * 60)
            print("AUTOENCODER")
            print("=" * 60)
            print(f"Device            : {DEVICE}")
            print(f"Trainable params  : {params:,}")
            print(f"Input shape       : {self.input_shape}")
            print(f"Latent shape      : {self.latent_shape}")
            print(f"Input elements    : {self.input_elements}")
            print(f"Latent elements   : {self.latent_elements}")
            print(f"Compression ratio : {self.compression_ratio:.2f}x")
            print("=" * 60)

            self._printed = True

        return reconstruction