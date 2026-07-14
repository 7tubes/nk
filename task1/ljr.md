# ljr 作业单：形态特征计算、评分分级与可视化复核

## 你的任务定位

你负责把 qyt 分割出的精子头部 mask 转换成可解释的形态学指标，并根据“拟合优度、长轴比、均匀度”完成量化评分和分级。最后还要把结果画成 overlay 图，方便人工复核。

简单说：你做的是“指标计算 + 评分分级 + 可视化验收”。

## 你需要完成的代码文件

建议你主要负责以下文件：

```text
nk/
  src/
    sperm_morphology/
      features.py
      scoring.py
      visualize.py
  scripts/
    review_results.py
  outputs/
    overlays/
    morphology_scores.csv
```

其中 `morphology_scores.csv` 由 dzh 的批处理主流程统一保存，但字段设计和内容由你来确定。

## 作业 1：实现轮廓和椭圆拟合 `features.py`

你需要从 qyt 输出的 `head_mask` 中提取轮廓，并拟合椭圆。

建议实现：

```python
def find_head_contour(head_mask):
    """找到头部 mask 的最大外轮廓。"""

def fit_head_ellipse(contour):
    """拟合头部椭圆。"""
```

必须处理：

1. mask 为空：返回失败原因 `empty_mask`。
2. 没有轮廓：返回失败原因 `no_contour`。
3. 轮廓点数小于 5：返回失败原因 `not_enough_contour_points`。
4. 椭圆拟合失败：返回失败原因 `ellipse_fit_failed`。

椭圆信息建议统一为：

```python
{
    "center": [cx, cy],
    "major_axis": L,
    "minor_axis": W,
    "angle": theta,
}
```

注意：

1. OpenCV `fitEllipse` 返回的两个轴不一定已经按长短排序，你要自己保证 `L >= W`。
2. 如果 `W` 接近 0，不能直接计算 `L / W`，要返回失败或加 `eps`。

预期效果：

1. 对每个成功 mask 都能得到头部中心、长轴、短轴、角度。
2. 后续 L、W、R 的计算稳定。

## 作业 2：计算基础形态指标

你需要实现：

```python
def compute_basic_features(head_mask, ellipse_info, config: dict) -> dict:
    """计算 L、W、R、HA、HD 等基础指标。"""
```

指标定义：

```text
L = major_axis
W = minor_axis
R = L / W
HA = head_mask 中前景像素数量
HD = sqrt(4 * HA / pi)
```

字段建议：

```python
{
    "L_px": L,
    "W_px": W,
    "R": R,
    "HA_px2": HA,
    "HD_px": HD,
}
```

如果 `config["calibration"]["um_per_pixel"]` 不为 `null`，再额外输出：

```python
{
    "L_um": L * um_per_pixel,
    "W_um": W * um_per_pixel,
    "HA_um2": HA * um_per_pixel * um_per_pixel,
    "HD_um": HD * um_per_pixel,
}
```

预期效果：

1. 没有显微镜标定时，所有长度保留像素单位。
2. 有显微镜标定时，自动补充微米单位。
3. 不伪造单位，不把像素当微米写。

## 作业 3：计算 SAS 和 LAS

你需要计算：

```text
SAS = 短轴对称性
LAS = 长轴对称性
```

建议第一版使用面积差近似，保证先跑通：

```text
LAS = 1 - abs(area_upper - area_lower) / (area_upper + area_lower + eps)
SAS = 1 - abs(area_left - area_right) / (area_left + area_right + eps)
```

更推荐的升级版本是镜像 IoU：

```text
LAS = IoU(mask, reflect(mask, about_long_axis))
SAS = IoU(mask, reflect(mask, about_short_axis))
```

你需要实现：

```python
def compute_symmetry_features(head_mask, ellipse_info, config: dict) -> dict:
    """计算 SAS 和 LAS。"""
```

实现步骤建议：

1. 根据椭圆角度 `theta`，把 mask 旋转到椭圆主轴坐标系。
2. 在旋转后的坐标系中：
   - x 方向代表长轴方向；
   - y 方向代表短轴方向。
3. 以中心为界，比较上下两半面积，得到 LAS。
4. 比较左右两半面积，得到 SAS。
5. 所有结果限制在 `0-1`。

预期效果：

1. 头部越对称，SAS 和 LAS 越接近 1。
2. 头部一侧缺损、空泡明显、形状偏斜时，对称性下降。

## 作业 4：计算拟合优度和均匀度

你需要实现：

```python
def compute_fit_goodness(head_mask, ellipse_info, config: dict) -> dict:
    """计算 head mask 和拟合椭圆 mask 的 IoU。"""

def compute_uniformity(head_mask, roi_pre, symmetry_features, config: dict) -> dict:
    """计算灰度均匀度和综合均匀度。"""
```

拟合优度：

```text
fit_iou = area(head_mask ∩ ellipse_mask) / area(head_mask ∪ ellipse_mask)
```

灰度均匀度：

```text
gray_uniformity = 1 - std(gray_values_in_head) / (mean(gray_values_in_head) + eps)
```

综合均匀度：

```text
uniformity = 0.6 * gray_uniformity + 0.4 * min(SAS, LAS)
```

注意：

1. `gray_uniformity` 可能小于 0，需要 clamp 到 `0-1`。
2. 如果 head mask 内像素为空，返回失败原因。
3. `ellipse_mask` 尺寸必须和 `head_mask` 一样。

预期效果：

1. 拟合优度能反映头部是否接近椭圆。
2. 均匀度能反映头部内部灰度是否稳定，以及形状是否对称。

## 作业 5：实现总特征函数

最终给 dzh 主流程提供：

```python
def compute_features(head_mask, roi_pre, config: dict) -> dict:
    """
    输入：
        head_mask: qyt 输出的头部二值 mask
        roi_pre: qyt 输出的预处理 ROI
        config: 配置

    输出：
        features: 全部形态指标
    """
```

成功时输出字段至少包含：

```text
L_px
W_px
R
SAS
LAS
HA_px2
HD_px
fit_iou
gray_uniformity
uniformity
```

失败时输出：

```python
{
    "success": False,
    "reason": "ellipse_fit_failed",
}
```

预期效果：

1. 所有评分所需指标都能从一个函数拿到。
2. 字段名稳定，dzh 写 CSV 时不会乱。

## 作业 6：实现评分与分级 `scoring.py`

你需要根据三个评分维度完成分级：

```text
拟合优度
长轴比
均匀度
```

建议实现：

```python
def score_fit_goodness(fit_iou, config: dict) -> float:
    """拟合优度评分，0-100。"""

def score_axis_ratio(R, config: dict) -> float:
    """长轴比评分，0-100。"""

def score_uniformity(uniformity, config: dict) -> float:
    """均匀度评分，0-100。"""

def score_features(features: dict, config: dict) -> dict:
    """计算总分和等级。"""
```

评分公式：

```text
fit_score = clamp((fit_iou - 0.60) / (0.85 - 0.60), 0, 1) * 100
axis_score = clamp(1 - abs(R - 1.62) / 0.35, 0, 1) * 100
uniformity_score = clamp((uniformity - 0.60) / (0.85 - 0.60), 0, 1) * 100
```

总分：

```text
total_score = 0.40 * fit_score + 0.30 * axis_score + 0.30 * uniformity_score
```

硬性剔除条件：

```text
如果特征计算失败：Reject
如果 HA 太小或太大：Reject
如果 R 极端异常：Reject
如果 min(SAS, LAS) < 0.55：Reject
如果 fit_iou < 0.50：Reject
```

分级：

| 等级 | 分数 |
| --- | --- |
| A | `>= 85` |
| B | `70-85` |
| C | `55-70` |
| D | `< 55` |
| Reject | 硬性失败 |

预期效果：

1. 每个成功样本都有三个子分数、总分、等级。
2. 明显异常样本进入 Reject，不强行分 A/B/C/D。
3. 后续调阈值时，只改 `morphology.yaml`。

## 作业 7：设计 CSV 输出字段

你需要和 dzh 对齐 `morphology_scores.csv` 的字段。建议字段如下：

```text
image_id
image_path
target_id
bbox_x1
bbox_y1
bbox_x2
bbox_y2
L_px
W_px
R
SAS
LAS
HA_px2
HD_px
fit_iou
gray_uniformity
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

预期效果：

1. CSV 可以直接用于 Excel 查看。
2. 每一行都能追溯到原图、mask 和 overlay。
3. 后续写论文或阶段汇报时，可以直接统计 A/B/C/D 数量。

## 作业 8：实现可视化 `visualize.py`

你需要把每个候选精子的结果画出来。

建议实现：

```python
def save_overlay(image, roi_info, head_mask, features, scores, config: dict) -> str:
    """保存原图 overlay，并返回 overlay_path。"""
```

overlay 图中至少画：

1. 原始 bbox。
2. ROI 范围。
3. 头部轮廓。
4. 拟合椭圆。
5. 长轴和短轴。
6. `grade` 和 `total_score`。

输出位置：

```text
outputs/overlays/
```

命名建议：

```text
0000_target0_A_91.2.png
0000_target1_Reject_bad_fit.png
```

预期效果：

1. 人工打开图就能判断评分是否合理。
2. 如果某个样本分数异常，可以快速看到是分割问题还是评分问题。

## 作业 9：实现复核脚本 `review_results.py`

你需要写一个复核辅助脚本。

建议功能：

1. 读取 `outputs/morphology_scores.csv`。
2. 按等级统计数量。
3. 随机抽取每个等级若干张 overlay。
4. 复制或列出这些 overlay 路径，供人工检查。
5. 统计 Reject 的原因分布。

运行方式建议：

```bash
python scripts/review_results.py --csv outputs/morphology_scores.csv --per-grade 20
```

预期效果：

1. 能快速知道 A/B/C/D/Reject 各有多少。
2. 能快速抽查每个等级是否靠谱。
3. 能发现主要失败原因，比如 `no_valid_contour` 太多，说明要找 qyt 调分割。

## 作业 10：你需要交付的结果

你最终需要提交：

1. `src/sperm_morphology/features.py`
2. `src/sperm_morphology/scoring.py`
3. `src/sperm_morphology/visualize.py`
4. `scripts/review_results.py`
5. 示例 overlay 图。
6. 一份简短说明，写明评分公式、等级阈值、当前最容易误判的情况。

## 验收标准

你的任务完成后，应满足：

1. 输入一个 head mask，可以算出 L、W、R、SAS、LAS、HA、HD。
2. 可以计算 fit_iou、gray_uniformity、uniformity。
3. 可以输出 fit_score、axis_score、uniformity_score、total_score、grade。
4. 可以对异常样本输出 Reject 和 reject_reason。
5. 可以保存带轮廓、椭圆、长短轴和分数的 overlay 图。
6. 可以从 CSV 统计等级分布和失败原因。

## 和其他人的接口约定

你从 dzh/qyt 接收：

```python
head_mask
roi_pre
roi_info
sample
config
```

你返回给 dzh：

```python
features
scores
overlay_path
```

你需要提醒 dzh CSV 里必须保留：

```text
mask_path
overlay_path
reject_reason
```

这样后续人工复核才方便。

## 额外加分项

如果有时间，可以做：

1. 画一个等级分布柱状图。
2. 画 `R`、`fit_iou`、`uniformity` 的直方图。
3. 找出每个等级中最接近边界的样本，作为人工重点复核对象。
4. 对 A 级样本按总分排序，输出前 20 个候选精子。
5. 写一个小报告：当前阈值下 A/B/C/D/Reject 的比例，以及主要失败原因。
