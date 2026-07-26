import torch
import torch.nn as nn

from config import (
    IN_CHANNELS,
    BASE_CHANNELS,
    LATENT_CHANNELS,
    NUM_RESIDUAL_BLOCKS,
)


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
                mode="nearest",
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

        self.enc0 = ConvBlock(
            IN_CHANNELS,
            BASE_CHANNELS,
        )

        self.enc1 = EncoderBlock(
            BASE_CHANNELS,
            BASE_CHANNELS * 2,
        )

        self.enc2 = EncoderBlock(
            BASE_CHANNELS * 2,
            BASE_CHANNELS * 4,
        )

        self.enc3 = EncoderBlock(
            BASE_CHANNELS * 4,
            LATENT_CHANNELS,
        )

        self.enc4 = EncoderBlock(
            LATENT_CHANNELS,
            LATENT_CHANNELS,
        )

        self.bottleneck = nn.Sequential(

            *[
                ResidualBlock(LATENT_CHANNELS)
                for _ in range(NUM_RESIDUAL_BLOCKS)
            ]

        )

        self.dec4 = Up(
            LATENT_CHANNELS,
            LATENT_CHANNELS,
        )

        self.dec3 = Up(
            LATENT_CHANNELS,
            BASE_CHANNELS * 4,
        )

        self.dec2 = Up(
            BASE_CHANNELS * 4,
            BASE_CHANNELS * 2,
        )

        self.dec1 = Up(
            BASE_CHANNELS * 2,
            BASE_CHANNELS,
        )

        self.head = nn.Conv2d(
            BASE_CHANNELS,
            IN_CHANNELS,
            kernel_size=1,
        )

    def forward(self, x):

        input_tensor = x

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
        reconstruction = reconstruction + input_tensor

        return reconstruction