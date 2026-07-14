# dzh 作业单：数据读取、ROI 裁剪与批处理主流程

## 你的任务定位

你负责把整个精子形态筛选项目的“地基”搭起来：让程序能稳定读取数据集、解析标签、裁剪 ROI、组织输出目录，并把后续 `qyt` 的分割模块和 `ljr` 的特征评分模块串成一个可批量运行的流程。

简单说：你做的是“数据入口 + 工程骨架 + 批处理流水线”。你的代码跑通以后，其他两个人的模块才能接进来。

## 你需要完成的代码文件

建议你主要负责以下文件：

```text
nk/
  configs/
    morphology.yaml
  src/
    sperm_morphology/
      __init__.py
      dataset.py
      crop.py
      batch_run.py
  scripts/
    run_morphology_screening.py
  outputs/
    rois/
    masks/
    overlays/
    morphology_scores.csv
    failed_cases.csv
```

其中 `outputs/` 不需要手动放结果文件，但程序运行时要能自动创建这些目录。

## 作业 1：建立项目基础目录

你需要在 `nk` 下建立下面的目录结构：

```text
configs/
src/sperm_morphology/
scripts/
outputs/
outputs/rois/
outputs/masks/
outputs/overlays/
```

预期效果：

1. 目录结构清晰，别人一看就知道代码、配置、脚本、结果分别在哪里。
2. 后续任何人运行脚本时，如果 `outputs` 不存在，程序可以自动创建。
3. 不要把运行结果、缓存文件、临时调试图混在源码目录里。

## 作业 2：编写配置文件 `configs/morphology.yaml`

你需要写一个统一配置文件，至少包含：

```yaml
data:
  image_dir: "../SpermTracking/ImagesWithLabels/images"
  label_dir: "../SpermTracking/ImagesWithLabels/labels"
  output_dir: "outputs"

calibration:
  um_per_pixel: null

crop:
  margin_px: 16
  min_box_width: 4
  min_box_height: 4

preprocess:
  clahe_clip_limit: 2.0
  clahe_tile_grid_size: 8
  gaussian_kernel: 3
  background_blur_kernel: 31

segmentation:
  min_head_area_px: 8
  max_head_area_px: 800
  min_contour_points: 5
  use_adaptive_threshold: true

scoring:
  weights:
    fit_goodness: 0.40
    axis_ratio: 0.30
    uniformity: 0.30
  axis_ratio_target: 1.62
  axis_ratio_tolerance: 0.35
  fit_iou_good: 0.85
  fit_iou_bad: 0.60
  uniformity_good: 0.85
  uniformity_bad: 0.60
  grade_thresholds:
    A: 85
    B: 70
    C: 55
```

预期效果：

1. 后续修改路径、阈值、裁剪边距时，只改配置文件，不改代码。
2. `qyt` 和 `ljr` 都能直接从这个配置文件里读取参数。
3. 配置里的路径必须相对于 `nk` 可以正常运行。

## 作业 3：实现 `dataset.py`

你的 `dataset.py` 负责读取图像和标签。当前标签格式是：

```text
id x1 y1 x2 y2
```

不是 YOLO 的归一化格式。

你需要实现这些函数：

```python
def load_config(config_path: str) -> dict:
    """读取 yaml 配置。"""

def list_image_label_pairs(config: dict) -> list[dict]:
    """遍历 images，找到同名 labels。"""

def parse_label_file(label_path: str) -> list[dict]:
    """解析单个 txt 标签文件。"""

def load_dataset(config: dict) -> list[dict]:
    """返回所有候选精子的样本列表。"""
```

每个样本建议返回：

```python
{
    "image_id": "0000",
    "image_path": ".../images/0000.jpg",
    "label_path": ".../labels/0000.txt",
    "target_id": 0,
    "bbox": [x1, y1, x2, y2],
}
```

必须处理的异常情况：

1. 图像存在但标签文件不存在：记录失败原因 `missing_label`。
2. 标签文件为空：跳过，不报错。
3. 标签行字段不足 5 个：记录失败原因 `bad_label_format`。
4. 坐标不是数字：记录失败原因 `bad_coordinate`。
5. `x2 <= x1` 或 `y2 <= y1`：记录失败原因 `invalid_bbox`。

预期效果：

1. 能读取 `SpermTracking/ImagesWithLabels/images` 和 `labels`。
2. 能生成一个候选目标列表。
3. 遇到坏标签不会让整个程序崩掉。

## 作业 4：实现 `crop.py`

你负责根据 bbox 裁剪 ROI。

需要实现：

```python
def clamp_bbox(bbox, image_width: int, image_height: int) -> list[int]:
    """把 bbox 限制在图像范围内。"""

def expand_bbox(bbox, margin_px: int, image_width: int, image_height: int) -> list[int]:
    """在 bbox 四周扩展 margin。"""

def crop_roi(image, bbox, config: dict) -> dict:
    """裁剪 ROI，并返回原图坐标和局部坐标映射。"""
```

`crop_roi` 返回格式：

```python
{
    "roi": roi_image,
    "roi_bbox_global": [rx1, ry1, rx2, ry2],
    "target_bbox_local": [x1-rx1, y1-ry1, x2-rx1, y2-ry1],
}
```

必须注意：

1. 标签坐标是浮点数，裁剪前要转成整数。
2. `x1 y1` 建议向下取整，`x2 y2` 建议向上取整。
3. 扩展后的 ROI 不能超过图像边界。
4. 如果 bbox 宽度或高度太小，要返回失败原因 `bbox_too_small`。

预期效果：

1. 每个候选精子都能被裁剪成一个 ROI。
2. ROI 周围有足够留白，方便 `qyt` 做头部分割。
3. 后续 `ljr` 可把局部坐标结果画回原图。

## 作业 5：实现批处理主流程 `batch_run.py`

你负责写一个主流程，把三个人的模块串起来。第一版可以先用占位函数，等 `qyt` 和 `ljr` 完成后再替换。

建议主流程结构：

```python
def run_batch(config_path: str):
    config = load_config(config_path)
    samples = load_dataset(config)
    ensure_output_dirs(config)

    rows = []
    failed_rows = []

    for sample in samples:
        image = read_image(sample["image_path"])
        roi_info = crop_roi(image, sample["bbox"], config)

        # qyt 负责
        roi_pre = preprocess_roi(roi_info["roi"], config)
        mask, quality_info = segment_head(
            roi_pre,
            roi_info["target_bbox_local"],
            config,
        )

        if mask is None:
            failed_rows.append(...)
            continue

        # ljr 负责
        features = compute_features(mask, roi_pre, config)
        scores = score_features(features, config)
        save_overlay(image, roi_info, mask, features, scores, config)

        rows.append(...)

    save_csv(rows, "outputs/morphology_scores.csv")
    save_csv(failed_rows, "outputs/failed_cases.csv")
```

你要负责的部分：

1. `ensure_output_dirs(config)`。
2. `read_image(image_path)`。
3. 主循环结构。
4. 保存 CSV。
5. 失败样例的统一记录格式。

预期效果：

1. 运行一次脚本，可以遍历完整数据集。
2. 即使某个样本失败，也不会中断整个批处理。
3. 最终一定能输出 `morphology_scores.csv` 和 `failed_cases.csv`。

## 作业 6：实现运行脚本

在 `scripts/run_morphology_screening.py` 中写命令行入口。

运行方式建议为：

```bash
python scripts/run_morphology_screening.py --config configs/morphology.yaml
```

预期效果：

1. 组员不需要打开源码，只需要运行脚本。
2. 能在终端打印处理进度，例如：

```text
[1/124] 0000 target=0 success grade=A
[2/124] 0000 target=1 failed reason=no_valid_contour
```

## 作业 7：你需要交付的结果

你最终需要提交：

1. `configs/morphology.yaml`
2. `src/sperm_morphology/dataset.py`
3. `src/sperm_morphology/crop.py`
4. `src/sperm_morphology/batch_run.py`
5. `scripts/run_morphology_screening.py`
6. 一份简短运行说明，可写在 `README.md` 或单独写 `task1_dzh_notes.md`

## 验收标准

你的任务完成后，应满足：

1. 能成功读取至少 `0000.jpg` 和 `0000.txt`。
2. 能解析出标签中的多个候选目标。
3. 能对每个候选目标裁剪 ROI。
4. 能自动创建 `outputs/rois`、`outputs/masks`、`outputs/overlays`。
5. 能生成 CSV，即使后续分割和评分模块还没完全做好，也能用占位结果跑通流程。
6. 坏数据不会让程序崩溃，而是写进 `failed_cases.csv`。

## 和其他人的接口约定

你需要给 `qyt` 提供：

```python
roi_info["roi"]
roi_info["target_bbox_local"]
config
```

你需要接收 `qyt` 返回：

```python
mask, quality_info
```

你需要给 `ljr` 提供：

```python
mask
roi_pre
roi_info
sample
config
```

你需要接收 `ljr` 返回：

```python
features
scores
overlay_path
```

## 额外加分项

如果有时间，可以做：

1. 随机保存前 20 个 ROI 到 `outputs/rois/`，方便大家检查裁剪是否正确。
2. 在 CSV 里加入 `roi_path` 字段。
3. 加一个 `--limit 20` 参数，方便小规模调试。
4. 统计总样本数、成功数、失败数、失败原因分布。
