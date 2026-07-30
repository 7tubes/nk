import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ROOT = Path(__file__).resolve().parent


class ResnetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, bias=False),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=64, n_blocks=9):
        super().__init__()
        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, base_channels, kernel_size=7, bias=False),
            nn.InstanceNorm2d(base_channels),
            nn.ReLU(True),
        ]

        channels = base_channels
        for _ in range(2):
            layers += [
                nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(channels * 2),
                nn.ReLU(True),
            ]
            channels *= 2

        for _ in range(n_blocks):
            layers.append(ResnetBlock(channels))

        for _ in range(2):
            layers += [
                nn.ConvTranspose2d(
                    channels,
                    channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                ),
                nn.InstanceNorm2d(channels // 2),
                nn.ReLU(True),
            ]
            channels //= 2

        layers += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_channels, out_channels, kernel_size=7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


def list_inputs(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXTS)


def get_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_image(tensor, grayscale=False):
    tensor = tensor.squeeze(0).detach().cpu().clamp(-1, 1)
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    image = transforms.ToPILImage()(tensor)
    if grayscale:
        image = image.convert("L").convert("RGB")
    return image


def pad_to_multiple(x, multiple=4):
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def load_generator(checkpoint_path, device, base_channels=None, res_blocks=None):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    train_args = checkpoint.get("args", {})
    base_channels = base_channels or int(train_args.get("base_channels", 64))
    res_blocks = res_blocks or int(train_args.get("res_blocks", 9))
    generator = ResnetGenerator(base_channels=base_channels, n_blocks=res_blocks).to(device)
    generator.load_state_dict(checkpoint["G_stain2transparent"])
    generator.eval()
    return generator


def parse_args():
    parser = argparse.ArgumentParser(description="Translate stained microscopy images to transparent/DIC-like style.")
    parser.add_argument("--input", type=str, default=str(ROOT / "datasets" / "stain_heads"), help="image file or folder")
    parser.add_argument("--checkpoint", type=str, default=str(ROOT / "checkpoints" / "latest.pt"))
    parser.add_argument("--output", type=str, default=str(ROOT / "outputs"))
    parser.add_argument("--suffix", type=str, default="_transparent")
    parser.add_argument("--grayscale", action="store_true", help="force RGB output to grayscale")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--base-channels", type=int, default=None)
    parser.add_argument("--res-blocks", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    device = get_device(args.device)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    generator = load_generator(args.checkpoint, device, args.base_channels, args.res_blocks)
    preprocess = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    inputs = list_inputs(args.input)
    if not inputs:
        raise FileNotFoundError(f"No input images found: {args.input}")

    for image_path in inputs:
        original = Image.open(image_path).convert("RGB")
        tensor = preprocess(original).unsqueeze(0).to(device)
        padded, (pad_h, pad_w) = pad_to_multiple(tensor, multiple=4)
        fake = generator(padded)
        if pad_h:
            fake = fake[:, :, :-pad_h, :]
        if pad_w:
            fake = fake[:, :, :, :-pad_w]

        out_image = tensor_to_image(fake, grayscale=args.grayscale)
        out_path = output_dir / f"{image_path.stem}{args.suffix}.png"
        out_image.save(out_path)
        print(out_path)


if __name__ == "__main__":
    main()
