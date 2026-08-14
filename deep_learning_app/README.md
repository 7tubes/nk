# 深度学习精子检测与可视化模块

本目录是项目中的 YOLO 精子目标检测与 Streamlit 可视化入口。它负责在显微图像中检测精子目标框，并可在界面中继续调用形态筛选系统完成红/黄/绿评分。

本次只扩展了可视化端与形态筛选的衔接，没有修改原有精子检测模型权重的默认调用位置。

## 目录结构

```text
deep_learning_app/
├── app.py
├── train_yolo.py
├── dataset.yaml
├── src/
│   └── dataset_utils.py
├── tests/
│   └── test_dataset_utils.py
├── dataset/
└── runs/
    └── sperm_detection/
        └── weights/
            ├── best.pt
            └── last.pt
```

`dataset/`、`runs/` 和本地 `.pt` 权重已经在 `.gitignore` 中忽略，不需要上传到合作库。

## 检测模型

可视化模块默认加载：

```text
deep_learning_app/runs/sperm_detection/weights/best.pt
```

这仍然是之前精子识别模型的默认权重位置。如果本机没有该文件，需要先由已有训练结果补齐；本次深度头部分割优化不改变这条检测模型链路。

## 数据集配置

`dataset.yaml` 使用模块内相对路径：

```yaml
path: "dataset"
train: images/train
val: images/val
test: images/val
nc: 1
names:
  - sperm
```

训练数据放入 `deep_learning_app/dataset/`，该目录不上传仓库。

## 训练检测模型

在项目根目录运行：

```powershell
python deep_learning_app\train_yolo.py --prepare-only
python deep_learning_app\train_yolo.py --epochs 100 --imgsz 768 --batch 4 --name sperm_detection
```

训练完成后，权重会输出到：

```text
deep_learning_app/runs/sperm_detection/weights/best.pt
deep_learning_app/runs/sperm_detection/weights/last.pt
```

通常使用 `best.pt` 做检测。

## 启动可视化与评分

在项目根目录运行：

```powershell
streamlit run deep_learning_app\app.py
```

界面功能：

- 从 `head_segmentation_app/datasets/` 选择本地待测图片。
- 或通过上传控件临时上传图片。
- 使用原 YOLO 检测模型检测精子目标框。
- 当 `configs/morphology.yaml` 中 `deep_segmentation.enabled: true` 且头部分割权重存在时，形态筛选会优先使用深度学习 mask。
- 在同一张图中显示每个精子的红/黄/绿评分结果。
- 输出 overlay 图和 CSV 到 `outputs/streamlit_screening/`。

## 深度头部分割配合

深度头部分割模块位于：

```text
head_segmentation_app/
```

可视化评分依赖 `configs/morphology.yaml` 中的配置：

```yaml
deep_segmentation:
  enabled: true
  model_path: "head_segmentation_app/runs/sperm_head_seg/weights/best.pt"
  conf: 0.15
  imgsz: 640
  max_det: 8
  fallback_to_traditional: true
```

其中 `model_path` 是相对项目根目录的路径。头部分割权重和测试图片不上传仓库，由每位成员在本机放到对应目录。

## 测试

在项目根目录运行：

```powershell
python -m unittest discover deep_learning_app\tests
python -m unittest tests.test_detection_screening
```

如果当前环境缺少 `opencv-python`、`streamlit` 或 `ultralytics`，需要先安装：

```powershell
python -m pip install -r requirements.txt
```
