from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, widths: tuple[int, int, int, int] = (32, 64, 128, 256)) -> None:
        super().__init__()
        self.encoders = nn.ModuleList()
        current_channels = in_channels
        for width in widths:
            self.encoders.append(ConvBlock(current_channels, width))
            current_channels = width

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(widths[-1], widths[-1] * 2)

        decoder_in_channels = widths[-1] * 2
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for width in reversed(widths):
            self.upconvs.append(nn.ConvTranspose2d(decoder_in_channels, width, kernel_size=2, stride=2))
            self.decoders.append(ConvBlock(width * 2, width))
            decoder_in_channels = width

        self.classifier = nn.Conv2d(widths[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips), strict=True):
            x = upconv(x)
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)
            x = decoder(x)

        return self.classifier(x)


def create_segmentation_model(model_name: str, num_classes: int, in_channels: int = 3) -> nn.Module:
    normalized = model_name.strip().lower()
    aliases = {
        "unet": "unet",
        "u-net": "unet",
    }
    resolved = aliases.get(normalized)
    if resolved == "unet":
        return UNet(in_channels=in_channels, num_classes=num_classes)
    raise ValueError(f"Unsupported training model '{model_name}'. Supported models: unet")
