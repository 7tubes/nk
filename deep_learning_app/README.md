# 深度学习精子目标检测应用

本目录是项目中的深度学习检测部分，使用 **Ultralytics YOLOv8** 训练单类别目标检测模型，用于在显微图像或视频帧中检测精子位置。

当前任务类型是：

```text
目标检测 object detection
类别数量 1 类
类别名称 sperm
输出结果 精子检测框 bbox + 置信度
```

注意：这部分代码只负责检测图像中的精子目标框，不直接完成精子头部分割、形态评分或 A/B/C/D 分级。形态筛选部分在 `nk/src/sperm_morphology` 中实现。

## 目录结构

```text
deep_learning_app/
├── app.py
├── train_yolo.py
├── dataset.yaml
├── yolov8n.pt
├── src/
│   └── dataset_utils.py
├── tests/
│   └── test_dataset_utils.py
├── dataset/
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
└── runs/
    └── sperm_detection/
        └── weights/
            ├── best.pt
            └── last.pt
```

## 各文件用途

`train_yolo.py`

训练入口文件。它会先调用 `src/dataset_utils.py` 把原始标注转换成 YOLO 数据集格式，然后加载 YOLOv8 预训练权重进行训练。

默认训练策略偏向提高小目标检出率：

```text
模型：YOLOv8n
输入尺寸：768
训练轮数：100
batch：4
数据增强：较弱 mosaic、较小 scale，减少小目标被增强破坏
训练数据：默认使用 tile 切片
```

`src/dataset_utils.py`

数据集转换工具。主要功能：

- 读取原始图片和标签。
- 过滤没有标签文件的图片，避免把未标注精子误当作背景。
- 将原始标签格式转换成 YOLO 标签格式。
- 随机划分 train/val。
- 默认把原图切成重叠 tile，提升小目标在输入图中的相对尺寸。
- 生成 `deep_learning_app/dataset.yaml`。

`dataset.yaml`

YOLO 训练配置文件，告诉 Ultralytics 数据集位置、训练集路径、验证集路径和类别名称。

当前内容类似：

```yaml
path: "C:/Users/Lenovo/Desktop/\u5927\u521b/nk/deep_learning_app/dataset"
train: images/train
val: images/val
nc: 1
names:
  - sperm
```

`app.py`

Streamlit 可视化检测界面。训练完成后，它会加载：

```text
deep_learning_app/runs/sperm_detection/weights/best.pt
```

然后支持上传图片或视频进行检测。图片检测默认开启 tiled inference，更适合检测图像中的小精子目标。

`tests/test_dataset_utils.py`

简单单元测试，检查原始 bbox 标签能否正确转换为 YOLO 标签格式。

`yolov8n.pt`

YOLOv8n 预训练权重。训练不是从零开始，而是在这个预训练模型基础上继续微调。

`dataset/`

自动生成的 YOLO 格式训练数据目录。一般不需要手动编辑。

`runs/`

训练输出目录。训练后的模型、训练曲线、验证集预测图、结果 CSV 都保存在这里。

## 环境配置

推荐使用 Anaconda 创建独立环境。

建议 Python 版本：

```text
Python 3.10 或 Python 3.11
```

不推荐优先使用 Python 3.13，因为部分深度学习库在新版本 Python 上可能兼容性不如 3.10/3.11 稳定。

### 创建 conda 环境

在 Anaconda Prompt 或 VS Code 终端中运行：

```powershell
conda create -n nk-morphology python=3.11 -y
conda activate nk-morphology
```

### 安装依赖

进入 `nk` 目录：

```powershell
cd C:\Users\Lenovo\Desktop\大创\nk
python -m pip install -r requirements.txt
```

`requirements.txt` 中包含：

```text
numpy
opencv-python
PyYAML
streamlit
ultralytics
```

### 检查环境

```powershell
python -c "import cv2, yaml, streamlit, ultralytics; print('environment ok')"
```

如果输出：

```text
environment ok
```

说明依赖安装成功。

## 原始数据要求

当前代码默认从下面路径读取原始数据：

```text
SpermTracking/ImagesWithLabels/images
SpermTracking/ImagesWithLabels/labels
```

图片格式：

```text
.jpg
```

标签格式：

```text
target_id x1 y1 x2 y2
```

示例：

```text
0 541.3199 108.0878 550.4078 123.0162
1 796.8204 215.4610 806.3124 231.4820
```

这里第一列 `target_id` 是目标编号，不是类别编号。转换成 YOLO 格式时，所有目标都会统一写成类别 `0`，也就是 `sperm`。

YOLO 标签格式为：

```text
class_id x_center y_center width height
```

其中 `x_center y_center width height` 都是相对于图像宽高归一化后的 `0-1` 数值。

## 准备 YOLO 数据集

只准备数据集，不训练：

```powershell
cd C:\Users\Lenovo\Desktop\大创\nk
python deep_learning_app\train_yolo.py --prepare-only
```

默认会生成：

```text
deep_learning_app/dataset/images/train
deep_learning_app/dataset/images/val
deep_learning_app/dataset/labels/train
deep_learning_app/dataset/labels/val
deep_learning_app/dataset.yaml
```

当前转换策略默认开启 tile 切片：

```text
tile_size: 512
overlap: 0.25
```

这样做的原因是精子目标在原图中很小。如果直接用整张图训练，目标在 640 或 768 输入尺寸下仍然很小，容易漏检。切片后，每个精子在输入图中的相对面积变大，更利于模型学习。

如果不想切片，可以使用：

```powershell
python deep_learning_app\train_yolo.py --prepare-only --full-image
```

## 训练模型

### CPU 训练

在 `nk` 目录下运行：

```powershell
cd C:\Users\Lenovo\Desktop\大创\nk
python deep_learning_app\train_yolo.py --epochs 100 --imgsz 768 --batch 4 --name sperm_detection
```

CPU 训练会比较慢，但可以运行。

训练完成后，模型会保存到：

```text
deep_learning_app/runs/sperm_detection/weights/best.pt
deep_learning_app/runs/sperm_detection/weights/last.pt
```

一般使用 `best.pt` 做检测。

### GPU 训练

如果电脑有 NVIDIA GPU，并且 PyTorch 已经安装 CUDA 版本，可以运行：

```powershell
python deep_learning_app\train_yolo.py --epochs 150 --imgsz 960 --batch 8 --device 0 --name sperm_detection
```

如果显存不够，可以降低：

```text
batch
imgsz
```

例如：

```powershell
python deep_learning_app\train_yolo.py --epochs 150 --imgsz 768 --batch 4 --device 0 --name sperm_detection
```

### 常用训练参数

```text
--model        使用的 YOLO 权重，默认 yolov8n.pt
--epochs       训练轮数，默认 100
--imgsz        输入图像尺寸，默认 768
--batch        batch size，默认 4
--name         runs/ 下的训练结果目录名
--device       auto、cpu 或 GPU 编号，例如 0
--val-ratio    验证集比例，默认 0.2
--seed         随机种子，默认 20260719
--full-image   关闭 tile 切片，使用整图训练
--tile-size    tile 尺寸，默认 512
--overlap      tile 重叠比例，默认 0.25
--prepare-only 只准备数据集，不训练
```

## 查看训练结果

训练输出目录：

```text
deep_learning_app/runs/sperm_detection
```

重点文件：

```text
weights/best.pt
weights/last.pt
results.csv
results.png
BoxP_curve.png
BoxR_curve.png
BoxPR_curve.png
BoxF1_curve.png
confusion_matrix.png
val_batch0_pred.jpg
val_batch0_labels.jpg
```

关注指标：

```text
metrics/precision(B)
metrics/recall(B)
metrics/mAP50(B)
metrics/mAP50-95(B)
```

如果目标是“尽量检测出图像中大部分精子”，优先关注：

```text
recall
mAP50
```

`precision` 高表示误检少，`recall` 高表示漏检少。显微图像精子检测中，如果后续还有人工复核或传统算法筛选，可以适当接受一些误检，优先提高 recall。

## 启动检测 App

训练完成后运行：

```powershell
cd C:\Users\Lenovo\Desktop\大创\nk\deep_learning_app
streamlit run app.py
```

启动后浏览器会打开类似地址：

```text
http://localhost:8501
```

如果浏览器没有自动打开，就手动复制终端中的地址到浏览器。

App 会加载：

```text
deep_learning_app/runs/sperm_detection/weights/best.pt
```

如果没有这个文件，需要先训练模型。

## App 参数建议

界面中有几个重要参数：

`Confidence`

置信度阈值。越低，检出的目标越多，但误检也可能增加。

建议：

```text
想尽量多检出：0.05 - 0.15
想减少误检：0.25 - 0.50
```

`NMS IoU`

非极大值抑制阈值，用于合并重复检测框。

建议先保持：

```text
0.45
```

`Image size`

推理输入尺寸。小目标漏检多时可以提高。

建议：

```text
768 或 960
```

`Tiled inference`

切片推理。建议默认开启，因为精子目标很小。

`Tile size`

切片大小。建议：

```text
512
```

如果 CPU 推理太慢，可以关闭 `Tiled inference`，但漏检可能增加。

## 运行测试

在 `nk` 目录下：

```powershell
cd C:\Users\Lenovo\Desktop\大创\nk
python -m unittest discover deep_learning_app\tests
```

看到下面结果说明测试通过：

```text
Ran 1 test
OK
```

## 常见问题

### 1. 找不到数据集 images/val

如果出现类似：

```text
images not found
missing path ... images\val
```

先重新准备数据：

```powershell
python deep_learning_app\train_yolo.py --prepare-only
```

然后检查：

```powershell
python -c "from ultralytics.data.utils import check_det_dataset; d=check_det_dataset('deep_learning_app/dataset.yaml'); print(d['train']); print(d['val'])"
```

### 2. 检测框太少

在 App 中降低：

```text
Confidence
```

例如调到：

```text
0.05 - 0.10
```

并开启：

```text
Tiled inference
```

### 3. 误检太多

提高：

```text
Confidence
```

例如调到：

```text
0.25 - 0.50
```

或者增加训练数据、检查标签质量。

### 4. CPU 训练太慢

可以先降低训练参数：

```powershell
python deep_learning_app\train_yolo.py --epochs 50 --imgsz 640 --batch 4 --name sperm_detection
```

但精度可能低于默认推荐配置。

如果需要共享模型权重，建议单独使用 GitHub Releases、网盘或 Git LFS。
