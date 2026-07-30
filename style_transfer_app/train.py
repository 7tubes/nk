import argparse
import itertools
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import save_image
from tqdm import tqdm


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
ROOT = Path(__file__).resolve().parent


def list_images(root):
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS]
    return sorted(files)


def find_yolo_label(image_path, image_root, label_root):
    if label_root is None:
        return None
    image_path = Path(image_path)
    image_root = Path(image_root)
    label_root = Path(label_root)
    try:
        relative = image_path.relative_to(image_root)
        label_path = label_root / relative.with_suffix(".txt")
    except ValueError:
        label_path = label_root / f"{image_path.stem}.txt"
    return label_path if label_path.exists() else None


def read_yolo_boxes(label_path):
    if label_path is None:
        return []
    boxes = []
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                _, cx, cy, bw, bh = parts[:5]
                boxes.append(tuple(float(v) for v in (cx, cy, bw, bh)))
            except ValueError:
                continue
    return boxes


def random_crop_params(width, height, crop_size):
    if width < crop_size or height < crop_size:
        raise ValueError(f"image is smaller than crop size: {width}x{height} < {crop_size}")
    left = random.randint(0, width - crop_size)
    top = random.randint(0, height - crop_size)
    return left, top


def label_centered_crop_params(width, height, crop_size, boxes):
    if not boxes:
        return random_crop_params(width, height, crop_size)
    cx, cy, _, _ = random.choice(boxes)
    center_x = cx * width
    center_y = cy * height
    jitter = crop_size * 0.25
    left = round(center_x - crop_size / 2 + random.uniform(-jitter, jitter))
    top = round(center_y - crop_size / 2 + random.uniform(-jitter, jitter))
    left = max(0, min(left, width - crop_size))
    top = max(0, min(top, height - crop_size))
    return left, top


def get_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def denorm(x):
    return (x * 0.5 + 0.5).clamp(0, 1)


class UnpairedMicroscopyDataset(Dataset):
    def __init__(
        self,
        stain_root,
        transparent_root,
        load_size,
        crop_size,
        transparent_label_root=None,
        label_crop_prob=0.85,
    ):
        self.stain_paths = list_images(stain_root)
        self.transparent_paths = list_images(transparent_root)
        self.stain_root = Path(stain_root)
        self.transparent_root = Path(transparent_root)
        self.transparent_label_root = Path(transparent_label_root) if transparent_label_root else None
        self.load_size = load_size
        self.crop_size = crop_size
        self.label_crop_prob = label_crop_prob
        if not self.stain_paths:
            raise FileNotFoundError(f"No stain images found in {stain_root}")
        if not self.transparent_paths:
            raise FileNotFoundError(f"No transparent images found in {transparent_root}")

        self.transparent_labels = [
            find_yolo_label(path, self.transparent_root, self.transparent_label_root)
            for path in self.transparent_paths
        ]
        self.num_transparent_labels = sum(path is not None for path in self.transparent_labels)

    def transform_image(self, image, boxes=None, prefer_label=False):
        image = TF.resize(
            image,
            self.load_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
        )
        width, height = image.size
        if prefer_label and boxes and random.random() < self.label_crop_prob:
            left, top = label_centered_crop_params(width, height, self.crop_size, boxes)
        else:
            left, top = random_crop_params(width, height, self.crop_size)
        image = TF.crop(image, top, left, self.crop_size, self.crop_size)
        if random.random() < 0.5:
            image = TF.hflip(image)
        tensor = TF.to_tensor(image)
        return TF.normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return max(len(self.stain_paths), len(self.transparent_paths))

    def __getitem__(self, index):
        stain_path = self.stain_paths[index % len(self.stain_paths)]
        transparent_index = random.randrange(len(self.transparent_paths))
        transparent_path = self.transparent_paths[transparent_index]
        label_path = self.transparent_labels[transparent_index]
        stain = Image.open(stain_path).convert("RGB")
        transparent = Image.open(transparent_path).convert("RGB")
        boxes = read_yolo_boxes(label_path)
        return {
            "stain": self.transform_image(stain),
            "transparent": self.transform_image(transparent, boxes=boxes, prefer_label=True),
            "stain_path": str(stain_path),
            "transparent_path": str(transparent_path),
        }


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


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64):
        super().__init__()

        def block(cin, cout, stride, norm=True):
            layers = [nn.Conv2d(cin, cout, kernel_size=4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.InstanceNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        self.model = nn.Sequential(
            *block(in_channels, base_channels, stride=2, norm=False),
            *block(base_channels, base_channels * 2, stride=2),
            *block(base_channels * 2, base_channels * 4, stride=2),
            *block(base_channels * 4, base_channels * 8, stride=1),
            nn.Conv2d(base_channels * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)


def init_weights(module):
    classname = module.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(module.weight.data, 0.0, 0.02)
        if getattr(module, "bias", None) is not None:
            nn.init.constant_(module.bias.data, 0.0)
    elif "InstanceNorm" in classname and getattr(module, "weight", None) is not None:
        nn.init.normal_(module.weight.data, 1.0, 0.02)
        nn.init.constant_(module.bias.data, 0.0)


class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.items = []

    def push_and_pop(self, batch):
        returned = []
        for image in batch.detach():
            image = image.unsqueeze(0)
            if len(self.items) < self.max_size:
                self.items.append(image)
                returned.append(image)
            elif random.random() > 0.5:
                index = random.randrange(len(self.items))
                old = self.items[index].clone()
                self.items[index] = image
                returned.append(old)
            else:
                returned.append(image)
        return torch.cat(returned, dim=0)


def make_lr_lambda(total_epochs, decay_epochs):
    def lr_lambda(epoch):
        if epoch < total_epochs:
            return 1.0
        progress = epoch - total_epochs
        return 1.0 - max(0, progress) / float(decay_epochs + 1)

    return lr_lambda


def save_checkpoint(path, epoch, models, optimizers, schedulers, args):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "G_stain2transparent": models["G_stain2transparent"].state_dict(),
            "G_transparent2stain": models["G_transparent2stain"].state_dict(),
            "D_stain": models["D_stain"].state_dict(),
            "D_transparent": models["D_transparent"].state_dict(),
            "optimizer_G": optimizers["G"].state_dict(),
            "optimizer_D_stain": optimizers["D_stain"].state_dict(),
            "optimizer_D_transparent": optimizers["D_transparent"].state_dict(),
            "scheduler_G": schedulers["G"].state_dict(),
            "scheduler_D_stain": schedulers["D_stain"].state_dict(),
            "scheduler_D_transparent": schedulers["D_transparent"].state_dict(),
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(path, models, optimizers=None, schedulers=None, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    models["G_stain2transparent"].load_state_dict(checkpoint["G_stain2transparent"])
    models["G_transparent2stain"].load_state_dict(checkpoint["G_transparent2stain"])
    models["D_stain"].load_state_dict(checkpoint["D_stain"])
    models["D_transparent"].load_state_dict(checkpoint["D_transparent"])
    if optimizers is not None:
        optimizers["G"].load_state_dict(checkpoint["optimizer_G"])
        optimizers["D_stain"].load_state_dict(checkpoint["optimizer_D_stain"])
        optimizers["D_transparent"].load_state_dict(checkpoint["optimizer_D_transparent"])
    if schedulers is not None:
        schedulers["G"].load_state_dict(checkpoint["scheduler_G"])
        schedulers["D_stain"].load_state_dict(checkpoint["scheduler_D_stain"])
        schedulers["D_transparent"].load_state_dict(checkpoint["scheduler_D_transparent"])
    return int(checkpoint.get("epoch", 0))


def save_samples(sample_dir, step, real_stain, fake_transparent, rec_stain, real_transparent):
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)
    grid = torch.cat(
        [
            denorm(real_stain[:4]).cpu(),
            denorm(fake_transparent[:4]).cpu(),
            denorm(rec_stain[:4]).cpu(),
            denorm(real_transparent[:4]).cpu(),
        ],
        dim=0,
    )
    save_image(grid, sample_dir / f"step_{step:07d}.jpg", nrow=min(4, real_stain.size(0)))


def parse_args():
    parser = argparse.ArgumentParser(description="Unpaired stained-to-transparent microscopy style transfer.")
    parser.add_argument("--data-root", type=str, default=str(ROOT / "datasets"))
    parser.add_argument("--stain-dir", type=str, default="stain")
    parser.add_argument("--transparent-dir", type=str, default="transparent")
    parser.add_argument("--transparent-label-dir", type=str, default="transparent_labels")
    parser.add_argument("--out-dir", type=str, default=str(ROOT / "checkpoints"))
    parser.add_argument("--epochs", type=int, default=100, help="epochs before linear LR decay")
    parser.add_argument("--decay-epochs", type=int, default=100, help="linear LR decay epochs")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--load-size", type=int, default=286)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0, help="use 0 on Windows unless you need more speed")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-cycle", type=float, default=10.0)
    parser.add_argument("--lambda-identity", type=float, default=5.0)
    parser.add_argument("--label-crop-prob", type=float, default=0.85)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--res-blocks", type=int, default=9)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--sample-every", type=int, default=100)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)
    data_root = Path(args.data_root)
    stain_root = data_root / args.stain_dir
    transparent_root = data_root / args.transparent_dir
    transparent_label_root = data_root / args.transparent_label_dir
    if not transparent_label_root.exists():
        transparent_label_root = None
    out_dir = Path(args.out_dir)

    dataset = UnpairedMicroscopyDataset(
        stain_root,
        transparent_root,
        args.load_size,
        args.crop_size,
        transparent_label_root=transparent_label_root,
        label_crop_prob=args.label_crop_prob,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    models = {
        "G_stain2transparent": ResnetGenerator(base_channels=args.base_channels, n_blocks=args.res_blocks).to(device),
        "G_transparent2stain": ResnetGenerator(base_channels=args.base_channels, n_blocks=args.res_blocks).to(device),
        "D_stain": PatchDiscriminator(base_channels=args.base_channels).to(device),
        "D_transparent": PatchDiscriminator(base_channels=args.base_channels).to(device),
    }
    for model in models.values():
        model.apply(init_weights)

    optimizer_G = torch.optim.Adam(
        itertools.chain(models["G_stain2transparent"].parameters(), models["G_transparent2stain"].parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    optimizer_D_stain = torch.optim.Adam(models["D_stain"].parameters(), lr=args.lr, betas=(0.5, 0.999))
    optimizer_D_transparent = torch.optim.Adam(models["D_transparent"].parameters(), lr=args.lr, betas=(0.5, 0.999))

    total_epochs = args.epochs + args.decay_epochs
    lr_lambda = make_lr_lambda(args.epochs, args.decay_epochs)
    schedulers = {
        "G": torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda),
        "D_stain": torch.optim.lr_scheduler.LambdaLR(optimizer_D_stain, lr_lambda=lr_lambda),
        "D_transparent": torch.optim.lr_scheduler.LambdaLR(optimizer_D_transparent, lr_lambda=lr_lambda),
    }
    optimizers = {"G": optimizer_G, "D_stain": optimizer_D_stain, "D_transparent": optimizer_D_transparent}

    start_epoch = 1
    if args.resume:
        loaded_epoch = load_checkpoint(args.resume, models, optimizers, schedulers, map_location=device)
        start_epoch = loaded_epoch + 1

    gan_loss = nn.MSELoss()
    cycle_loss = nn.L1Loss()
    identity_loss = nn.L1Loss()
    fake_stain_pool = ReplayBuffer()
    fake_transparent_pool = ReplayBuffer()
    global_step = (start_epoch - 1) * len(loader)

    print(f"Device: {device}")
    print(f"Stain images: {len(dataset.stain_paths)} | transparent images: {len(dataset.transparent_paths)}")
    print(f"Transparent label files: {dataset.num_transparent_labels}")
    print(f"Training settings: crop={args.crop_size}, load={args.load_size}, channels={args.base_channels}, blocks={args.res_blocks}")
    print(f"Training epochs: {start_epoch}..{total_epochs}")

    for epoch in range(start_epoch, total_epochs + 1):
        progress = tqdm(loader, desc=f"epoch {epoch}/{total_epochs}", leave=True)
        for batch in progress:
            real_stain = batch["stain"].to(device, non_blocking=True)
            real_transparent = batch["transparent"].to(device, non_blocking=True)

            optimizer_G.zero_grad(set_to_none=True)

            same_transparent = models["G_stain2transparent"](real_transparent)
            same_stain = models["G_transparent2stain"](real_stain)
            loss_id = (
                identity_loss(same_transparent, real_transparent)
                + identity_loss(same_stain, real_stain)
            ) * args.lambda_identity

            fake_transparent = models["G_stain2transparent"](real_stain)
            pred_fake_transparent = models["D_transparent"](fake_transparent)
            target_real = torch.ones_like(pred_fake_transparent)
            loss_g_stain2transparent = gan_loss(pred_fake_transparent, target_real)

            fake_stain = models["G_transparent2stain"](real_transparent)
            pred_fake_stain = models["D_stain"](fake_stain)
            loss_g_transparent2stain = gan_loss(pred_fake_stain, torch.ones_like(pred_fake_stain))

            rec_stain = models["G_transparent2stain"](fake_transparent)
            rec_transparent = models["G_stain2transparent"](fake_stain)
            loss_cycle = (
                cycle_loss(rec_stain, real_stain) + cycle_loss(rec_transparent, real_transparent)
            ) * args.lambda_cycle

            loss_G = loss_id + loss_g_stain2transparent + loss_g_transparent2stain + loss_cycle
            loss_G.backward()
            optimizer_G.step()

            optimizer_D_transparent.zero_grad(set_to_none=True)
            pred_real_transparent = models["D_transparent"](real_transparent)
            loss_d_real_transparent = gan_loss(pred_real_transparent, torch.ones_like(pred_real_transparent))
            pooled_fake_transparent = fake_transparent_pool.push_and_pop(fake_transparent)
            pred_fake_transparent = models["D_transparent"](pooled_fake_transparent.detach())
            loss_d_fake_transparent = gan_loss(pred_fake_transparent, torch.zeros_like(pred_fake_transparent))
            loss_D_transparent = 0.5 * (loss_d_real_transparent + loss_d_fake_transparent)
            loss_D_transparent.backward()
            optimizer_D_transparent.step()

            optimizer_D_stain.zero_grad(set_to_none=True)
            pred_real_stain = models["D_stain"](real_stain)
            loss_d_real_stain = gan_loss(pred_real_stain, torch.ones_like(pred_real_stain))
            pooled_fake_stain = fake_stain_pool.push_and_pop(fake_stain)
            pred_fake_stain = models["D_stain"](pooled_fake_stain.detach())
            loss_d_fake_stain = gan_loss(pred_fake_stain, torch.zeros_like(pred_fake_stain))
            loss_D_stain = 0.5 * (loss_d_real_stain + loss_d_fake_stain)
            loss_D_stain.backward()
            optimizer_D_stain.step()

            global_step += 1
            if args.sample_every > 0 and global_step % args.sample_every == 0:
                save_samples(out_dir / "samples", global_step, real_stain, fake_transparent, rec_stain, real_transparent)

            progress.set_postfix(
                G=f"{loss_G.item():.3f}",
                D_T=f"{loss_D_transparent.item():.3f}",
                D_S=f"{loss_D_stain.item():.3f}",
                lr=f"{schedulers['G'].get_last_lr()[0]:.2e}",
            )

        for scheduler in schedulers.values():
            scheduler.step()

        save_checkpoint(out_dir / "latest.pt", epoch, models, optimizers, schedulers, args)
        if args.save_every > 0 and (epoch % args.save_every == 0 or epoch == total_epochs):
            save_checkpoint(out_dir / f"epoch_{epoch:04d}.pt", epoch, models, optimizers, schedulers, args)

    print("Done.")


if __name__ == "__main__":
    main()
