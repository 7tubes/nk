# 精子图像风格转换模块

本目录是项目中新加入的精子显微图像风格转换部分，来源于之前实现的染色图像转透明/DIC 风格代码。当前融合方式是独立模块接入，不修改现有 `deep_learning_app` 目标检测、`src/sperm_morphology` 形态筛选和 `scripts` 批处理逻辑。

当前任务类型是：

```text
图像到图像风格转换 image-to-image translation
训练方式 非配对训练 unpaired training
模型结构 CycleGAN 风格双生成器 + 双判别器
输入 染色精子显微图像或染色头部 patch
输出 透明/DIC-like 风格图像
```

## 目录结构

```text
style_transfer_app/
├── train.py
├── infer.py
├── extract_stain_heads.py
├── requirements.txt
├── README.md
├── datasets/
│   ├── stain/
│   ├── stain_heads/
│   ├── stain_head_masks/
│   ├── stain_head_labels/
│   ├── transparent/
│   └── transparent_labels/
├── checkpoints/
└── outputs/
```

## 各文件用途

`train.py`

风格转换训练入口。它会读取染色域图片和透明/DIC 目标风格图片，使用非配对方式训练两个方向的生成器：

- `G_stain2transparent`：染色图像转换成透明/DIC 风格。
- `G_transparent2stain`：透明/DIC 风格转换回染色风格，用于 cycle consistency 约束。
- `D_transparent`：判断透明域图像真假。
- `D_stain`：判断染色域图像真假。

训练时同时使用 GAN loss、cycle loss 和 identity loss，保证输出风格接近目标域，同时尽量保留精子头部几何结构。

`infer.py`

推理入口。训练完成后加载 `checkpoints/latest.pt` 或指定 checkpoint，把单张图片或整个文件夹中的染色图像转换为透明/DIC-like 风格，并保存到 `outputs/`。

`extract_stain_heads.py`

染色头部 patch 自动提取工具。如果原始染色图片没有人工标注，可以先利用紫蓝染色区域提取精子头部 patch，生成训练用的 `stain_heads/`、`stain_head_masks/` 和 `stain_head_labels/`。

`requirements.txt`

风格转换模块的独立依赖说明。当前根目录 `requirements.txt` 也已经同步加入这些依赖，伙伴们可以直接在项目根目录统一安装。

## 数据集要求

本模块使用非配对训练，所以染色图片和透明/DIC 图片不需要一一对应。

推荐的数据摆放方式：

```text
style_transfer_app/datasets/
├── stain/
│   ├── stain_001.jpg
│   └── stain_002.jpg
├── stain_heads/
│   ├── stain_001_head_000.png
│   └── stain_001_head_001.png
├── stain_head_masks/
│   ├── stain_001_head_000.png
│   └── stain_001_head_001.png
├── stain_head_labels/
│   ├── stain_001_head_000.txt
│   └── stain_001_head_001.txt
├── transparent/
│   ├── 1.jpg
│   └── 2.jpg
└── transparent_labels/
    ├── 1.txt
    └── 2.txt
```

数据说明：

- `stain/`：原始染色精子显微图像。
- `stain_heads/`：从染色图像中提取出的头部 patch，推荐作为训练输入域。
- `stain_head_masks/`：与 `stain_heads/` 对齐的头部二值 mask，后续做分割训练时可以复用。
- `stain_head_labels/`：与 `stain_heads/` 对齐的 YOLO 框标注。
- `transparent/`：透明/DIC 或无染色风格显微图像。
- `transparent_labels/`：透明域图像的 YOLO 标注，可选但推荐提供。

`transparent_labels/` 使用 YOLO txt 格式：

```text
class x_center y_center width height
```

其中坐标均为 `0-1` 范围内的归一化值，文件名需要与透明域图片对应，例如 `transparent/1.jpg` 对应 `transparent_labels/1.txt`。

## 实现过程

### 1. 染色头部区域提取

如果 `datasets/stain/` 中是整张染色显微图，建议先运行：

```powershell
python style_transfer_app\extract_stain_heads.py --clean-output --max-total-heads 300 --max-heads-per-image 3 --shuffle --debug-dir style_transfer_app\datasets\stain_head_debug
```

主要处理流程：

- 对染色图像做灰世界白平衡、背景校正和去噪。
- 转换到 HSV、LAB 和 BGR 颜色空间。
- 根据紫蓝染色区域构造弱候选 mask 和高置信 seed mask。
- 只保留同时包含 seed 的弱候选连通域，减少背景碎片误检。
- 对候选区域做孔洞填充、闭运算和凸包修补。
- 裁剪成固定尺寸头部 patch，并同步保存 mask 和 YOLO label。

调试输出说明：

- `*_debug.jpg`：绿色框表示保留的头部 patch，黄色轮廓表示最终 mask，红色轮廓表示高置信 seed。
- `*_weak.png`：背景校正和去噪后的宽松候选区域。
- `*_seed.png`：更严格的染色核心区域。
- `*_score.png`：染色增强得分图。

如果杂质保留过多，可以提高阈值：

```powershell
python style_transfer_app\extract_stain_heads.py --clean-output --max-total-heads 300 --max-heads-per-image 3 --shuffle --seed-sat-min 55 --seed-score-min 75 --min-area 150 --debug-dir style_transfer_app\datasets\stain_head_debug
```

如果真实头部漏检较多，可以放宽阈值：

```powershell
python style_transfer_app\extract_stain_heads.py --clean-output --max-total-heads 300 --max-heads-per-image 3 --shuffle --seed-sat-min 30 --seed-score-min 45 --weak-sat-min 8 --weak-score-min 20 --min-area 50 --debug-dir style_transfer_app\datasets\stain_head_debug
```

### 2. 非配对风格转换训练

训练时默认使用 `stain_heads/` 作为染色输入域，使用 `transparent/` 作为目标风格域：

```powershell
python style_transfer_app\train.py --stain-dir stain_heads --device auto
```

默认路径已经绑定到 `style_transfer_app` 目录，因此在项目根目录运行也可以正确读取：

```text
style_transfer_app/datasets
style_transfer_app/checkpoints
```

如果要直接使用整张染色图训练，可以改为：

```powershell
python style_transfer_app\train.py --stain-dir stain --device auto
```

### 3. 风格转换推理

训练完成后运行：

```powershell
python style_transfer_app\infer.py --input style_transfer_app\datasets\stain_heads --checkpoint style_transfer_app\checkpoints\latest.pt --output style_transfer_app\outputs --grayscale
```

单张图片推理：

```powershell
python style_transfer_app\infer.py --input your_image.jpg --checkpoint style_transfer_app\checkpoints\latest.pt --output style_transfer_app\outputs --grayscale
```

输出文件默认命名为：

```text
原文件名_transparent.png
```

## 标准训练参数

当前模型默认参数已经调成更接近标准 CycleGAN 训练状态，适合伙伴们直接开始正式训练：

```text
epochs=100
decay_epochs=100
batch_size=1
load_size=286
crop_size=256
lr=0.0002
lambda_cycle=10.0
lambda_identity=5.0
base_channels=64
res_blocks=9
num_workers=0
device=auto
```

参数含义：

- `epochs`：保持初始学习率训练的轮数。
- `decay_epochs`：线性衰减学习率的轮数。
- `batch_size`：CycleGAN 经典配置为 1，显存压力较小。
- `load_size`：先把图像缩放到该尺寸。
- `crop_size`：再随机裁剪训练 patch。
- `lr`：Adam 初始学习率，标准值为 `2e-4`。
- `lambda_cycle`：循环一致性约束权重，保持结构不被明显改变。
- `lambda_identity`：identity 约束权重，减少颜色和亮度过度漂移。
- `base_channels`：生成器和判别器基础通道数。
- `res_blocks`：ResNet 生成器残差块数量。
- `num_workers=0`：Windows 下更稳定。

如果只是检查流程是否能跑通，可以用小参数快速测试：

```powershell
python style_transfer_app\train.py --stain-dir stain_heads --epochs 1 --decay-epochs 0 --base-channels 16 --res-blocks 2 --load-size 128 --crop-size 112 --sample-every 5 --device cpu
```

## 训练输出

训练结果保存在：

```text
style_transfer_app/checkpoints/
```

重点文件：

```text
latest.pt
epoch_0001.pt
epoch_0100.pt
samples/step_0000100.jpg
```

sample 图像按行展示：

```text
real stained -> generated transparent -> reconstructed stained -> real transparent
```

建议训练时重点观察：

- `generated transparent` 是否接近透明/DIC 风格。
- 精子头部轮廓是否被保留。
- 背景是否出现大面积伪影。
- 是否出现结构扭曲、头尾粘连或局部过度平滑。

## 与当前识别筛选系统的衔接方式

当前模块不会自动改变现有识别筛选流程。推荐衔接方式是先进行风格转换，再把转换后的图片作为检测或后续分割训练数据使用。

典型流程：

```text
染色原图
-> extract_stain_heads.py 提取染色头部 patch
-> train.py 训练染色到透明风格转换模型
-> infer.py 生成透明/DIC-like 图像
-> deep_learning_app 或 scripts 中的识别筛选流程继续处理
```

用于分割训练时，需要保持图像和 mask 文件名对齐：

```text
seg_dataset/
├── images/
│   ├── stain_001_head_000_transparent.png
│   └── stain_001_head_001_transparent.png
└── masks/
    ├── stain_001_head_000.png
    └── stain_001_head_001.png
```

注意不要对 mask 做风格转换。风格转换只处理图像，mask 继续使用原始头部 mask。

## 环境配置

推荐使用 Python 3.10 或 Python 3.11。

在项目根目录安装统一依赖：

```powershell
python -m pip install -r requirements.txt
```

或者只安装本模块依赖：

```powershell
python -m pip install -r style_transfer_app\requirements.txt
```

如果使用 NVIDIA GPU，建议先按照 PyTorch 官方说明安装匹配 CUDA 的 `torch` 和 `torchvision`，再安装项目依赖。

检查环境：

```powershell
python -c "import torch, torchvision, cv2, PIL, tqdm; print('style transfer environment ok')"
```

## 常见问题

### 1. 提示找不到 `stain_heads`

先确认是否已经运行头部提取：

```powershell
python style_transfer_app\extract_stain_heads.py --clean-output
```

也可以直接使用整张染色图训练：

```powershell
python style_transfer_app\train.py --stain-dir stain
```

### 2. 显存不足

优先降低：

```text
crop_size
load_size
base_channels
```

例如：

```powershell
python style_transfer_app\train.py --stain-dir stain_heads --load-size 256 --crop-size 224 --base-channels 48 --res-blocks 6 --device 0
```

### 3. 输出过灰或细节被抹掉

可以尝试：

- 增加透明域图像数量。
- 保证染色域和透明域显微倍率接近。
- 检查 `transparent_labels/` 是否能让目标风格 crop 更集中在精子头部。
- 适当降低 `lambda_identity`，例如 `--lambda-identity 3.0`。

### 4. 输出风格正常但形态变形

可以尝试：

- 增加 `lambda_cycle`，例如 `--lambda-cycle 12.0`。
- 提高训练数据中清晰头部样本占比。
- 使用 `stain_heads/` 而不是整张图训练，减少背景对模型的干扰。

