# 精子头部 YOLOv8-Seg 分割与评级测试模块

本目录是当前精子识别与形态筛选系统中的深度学习头部分割模块。之前的风格转换路线暂时停止，本模块改为直接使用 LabelMe polygon 头部标注训练 YOLOv8-Seg，生成精子头部 mask，然后继续调用现有形态特征计算和评分评级代码。

当前完整链路：

```text
LabelMe head polygon 标注
-> LabelMe 转 YOLO-Seg 标签
-> YOLOv8-Seg 训练头部分割模型
-> 模型推理生成头部 mask
-> compute_features() 提取形态特征
-> score_features() 输出 A/B/C/D/Reject 评级
```

## 数据放置规则

以后本模块所有本地测试图片和 LabelMe 标注都放在：

```text
head_segmentation_app/datasets/
```

这个目录已经写入 `.gitignore`，图片、JSON、视频等真实数据不会上传到合作仓库。仓库里只保留空目录占位文件：

```text
head_segmentation_app/datasets/.gitkeep
head_segmentation_app/datasets/images/.gitkeep
head_segmentation_app/datasets/labelme/.gitkeep
```

推荐但不强制的摆放方式：

```text
head_segmentation_app/datasets/
├── full_0002.jpg
├── full_0002.json
├── video_100.jpg
├── video_100.json
├── images/
│   └── 放只用于推理评级的普通图片
└── labelme/
    └── 放用于训练转换的 LabelMe 图片和 JSON
```

当前脚本默认直接读取 `head_segmentation_app/datasets/`，因此不再需要写绝对路径。

## 准备 YOLO-Seg 数据集

在项目根目录 `nk` 下运行：

```powershell
python head_segmentation_app\prepare_dataset.py
```

默认输入：

```text
head_segmentation_app/datasets/
```

默认输出：

```text
head_segmentation_app/dataset/
head_segmentation_app/dataset.yaml
```

转换逻辑：

- 读取同名图片和 LabelMe JSON。
- 只保留 `label == "head"` 的 polygon。
- 忽略 `tail` 标注。
- 输出 YOLOv8-Seg 标签：`class_id x1 y1 x2 y2 ...`。
- 默认使用 `tile_size=320` 和 `overlap=0.25` 做切片，便于模型学习小尺寸头部。

如果只想用某个子目录，例如 `datasets/labelme/`：

```powershell
python head_segmentation_app\prepare_dataset.py --source-dir head_segmentation_app\datasets\labelme
```

## 训练 YOLOv8-Seg

只准备数据、不训练：

```powershell
python head_segmentation_app\train_yolo_seg.py --prepare-only
```

正式训练：

```powershell
python head_segmentation_app\train_yolo_seg.py --model yolov8n-seg.pt --epochs 120 --imgsz 640 --batch 4 --name sperm_head_seg
```

训练完成后常用权重：

```text
head_segmentation_app/runs/sperm_head_seg/weights/best.pt
head_segmentation_app/runs/sperm_head_seg/weights/last.pt
```

## 直接测试“深度 mask -> 形态评级”

这是你现在最需要的测试入口。把待测图片放进 `head_segmentation_app/datasets/` 后，直接运行：

```powershell
python head_segmentation_app\infer_masks.py
```

它会默认：

- 从 `head_segmentation_app/datasets/` 递归读取图片。
- 加载 `head_segmentation_app/runs/sperm_head_seg/weights/best.pt`。
- 使用 YOLOv8-Seg 生成头部 mask。
- 保存 mask 和 overlay。
- 调用 `compute_features()` 提取形态特征。
- 调用 `score_features()` 输出评级。

输出位置：

```text
head_segmentation_app/outputs/masks/
head_segmentation_app/outputs/overlays/
head_segmentation_app/outputs/head_segmentation_features.csv
```

重点查看 CSV 中这些字段：

```text
image_id
head_id
confidence
HA_px2
L_px
W_px
R
SAS
LAS
fit_iou
uniformity
fit_score
axis_score
uniformity_score
total_score
grade
reject_reason
mask_path
overlay_path
```

如果只想测一张图片，用相对路径：

```powershell
python head_segmentation_app\infer_masks.py --input head_segmentation_app\datasets\full_0002.jpg
```

如果想测一个子目录：

```powershell
python head_segmentation_app\infer_masks.py --input head_segmentation_app\datasets\images
```

如果想用 `last.pt`：

```powershell
python head_segmentation_app\infer_masks.py --model head_segmentation_app\runs\sperm_head_seg\weights\last.pt
```

## 与现有筛选系统的区别

`infer_masks.py` 是“只测试头部分割模型和评级系统”的入口，不依赖精子检测框。它适合快速判断：

- 深度学习 mask 是否贴合头部。
- mask 进入 `features.py` 后能否算出稳定形态特征。
- `scoring.py` 的 A/B/C/D/Reject 评级是否合理。

完整识别筛选系统仍然使用：

```powershell
python scripts\run_detection_screening.py --image your_image.jpg
```

当 `configs/morphology.yaml` 中开启：

```yaml
deep_segmentation:
  enabled: true
  model_path: "head_segmentation_app/runs/sperm_head_seg/weights/best.pt"
```

完整系统会在检测框 ROI 内优先使用 YOLOv8-Seg mask，再进入原来的形态评分流程。

## 在可视化界面中查看评分

如果想打开项目可视化模块，在界面里选择图片并查看多个精子的打分情况，运行：

```powershell
cd deep_learning_app
streamlit run app.py
```

打开浏览器页面后：

- 在 `Open local image` 中选择 `head_segmentation_app/datasets/` 里的图片。
- 保持 `Morphology screening colors` 开启。
- 侧边栏会显示 `Deep head segmentation: enabled`，表示当前评级流程调用的是 YOLOv8-Seg 头部分割。
- 页面中会显示红/黄/绿评分图和每个精子的 `grade`、`total_score`、`reject_reason`、`fit_iou`、`R`、`uniformity`。

可视化模块会同时保存结果：

```text
outputs/streamlit_screening/
├── 图片名_overlay.png
└── 图片名_results.csv
```

## 常见判断

- `outputs/head_segmentation_features.csv` 只有表头：模型没有生成有效 mask，或输入目录下没有可读图片。
- overlay 中 mask 明显偏大或粘连：需要检查 LabelMe 标注质量，或后续提高形态筛选阈值。
- `grade` 大量为 `Reject`：通常是面积、长宽比、对称性或拟合优度不满足当前 `configs/morphology.yaml` 阈值，需要结合深度 mask 的统计结果重新校准。
- `confidence` 有输出但 `features.success=False`：说明模型输出了 mask，但 mask 轮廓不适合椭圆拟合或形态计算。

## 当前实现重点

- 数据入口改为模块内相对路径 `head_segmentation_app/datasets/`。
- 真实图片和标注数据已加入 `.gitignore`，不会上传合作库。
- `infer_masks.py` 已经不只是导出 mask，还会继续调用现有评级系统输出 `grade` 和 `total_score`。
- 训练、推理、评级测试都不再需要写外部绝对路径。
