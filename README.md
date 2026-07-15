working station：（y/n）
qyt：y
ljr：n
dzh：n

常用git代码：
git pull：把github上其他人更新的代码版本更新过来
git push：把自己的修改上传到仓库git

task：
1）精子筛选（高倍率）
2）多目标跟踪算法（操作倍率）
3）精子姿态检测与调整
4）精子吸取

日志：
7.6-7.17task: 精子筛选（高倍率）

7.14：qyt：
今天主要完成精子形态筛选任务中“ROI 预处理”和“阈值分割候选生成”两部分代码。我的工作重点是把 dzh 后续裁剪出的精子候选 ROI 图像，先处理成适合分割的增强灰度图，再基于增强后的 ROI 生成多个头部分割候选 mask，为后续连通域筛选、精子头部特征提取和形态评分做准备。

目前主要编写和整理了以下 3 个 Python 文件：

1）`nk/src/sperm_morphology/preprocess.py`

作用：
该文件负责 ROI 图像预处理，也就是把原始候选区域图像处理成更适合分割的灰度增强图。由于显微图像中可能存在背景亮度不均、噪声、对比度低等问题，如果直接做阈值分割，容易把背景颗粒或尾部误分成头部，所以需要先做预处理。

核心函数：
`preprocess_roi(roi_gray, config)`

输入：
- `roi_gray`：dzh 裁剪得到的单个精子候选 ROI，可以是灰度图，也可以是 BGR/BGRA 图。
- `config`：预处理参数字典，例如去噪方式、卷积核大小、背景校正方式、CLAHE 对比度增强参数等。

主要处理流程：
- 检查输入图像是否合法。
- 将输入图像统一转换为单通道灰度图。
- 将灰度值归一化到 `0-255`。
- 使用 Gaussian 或 median blur 去噪。
- 使用大核 Gaussian blur 估计背景亮度，并通过除法或减法进行背景校正。
- 使用 CLAHE 增强局部对比度，使精子头部边缘更明显。

输出：
- `enhanced_roi`：经过归一化、去噪、背景校正和 CLAHE 增强后的 `uint8` 单通道灰度图。

预期效果：
- 提高精子头部和背景之间的对比度。
- 减少显微图像背景不均对分割的影响。
- 为后续 Otsu 和 adaptive threshold 分割提供更稳定的输入。

2）`nk/src/sperm_morphology/segment_head.py`

作用：
该文件目前只完成任务二的内容：根据预处理后的 ROI 图像生成多个二值分割候选结果。因为不同图像中精子头部可能表现为偏亮或偏暗，单一阈值方法不稳定，所以同时生成多种候选 mask，后续再由连通域筛选模块选择最像头部的一个。

核心函数：
`threshold_candidates(roi_pre, config=None)`

输入：
- `roi_pre`：由 `preprocess_roi()` 输出的预处理后 ROI 灰度图。
- `config`：阈值分割参数字典，例如自适应阈值方法、邻域大小、常数 C、阈值前轻微平滑核大小等。

当前实现的 4 种分割候选：
- `otsu_binary`：Otsu 正向阈值，适合目标比背景更亮的情况。
- `otsu_binary_inv`：Otsu 反向阈值，适合目标比背景更暗的情况。
- `adaptive_binary`：自适应阈值正向，适合局部亮度不均且目标偏亮的情况。
- `adaptive_binary_inv`：自适应阈值反向，适合局部亮度不均且目标偏暗的情况。

输出：
返回一个列表，每个元素是一个字典：

```python
[
    {"method": "otsu_binary", "mask": otsu_binary},
    {"method": "otsu_binary_inv", "mask": otsu_binary_inv},
    {"method": "adaptive_binary", "mask": adaptive_binary},
    {"method": "adaptive_binary_inv", "mask": adaptive_binary_inv},
]
```

其中：
- `method` 表示该候选 mask 的生成方法。
- `mask` 是二值图像，像素值为 `0` 或 `255`。

预期效果：
- 对同一个 ROI 同时提供多种分割可能。
- 解决精子头部有时偏亮、有时偏暗的问题。
- 为后续“形态学清理”和“候选连通域筛选”提供输入。

3）`nk/src/sperm_morphology/utils.py`

作用：
该文件用于存放 `preprocess.py` 和 `segment_head.py` 中都会用到的公共工具函数，避免两个文件重复写同样的输入检查、灰度转换和卷积核处理代码。

目前包含两个函数：

`ensure_odd_kernel(value, default_value=3, minimum=1)`

输入：
- `value`：用户配置的卷积核大小。
- `default_value`：配置不合法时使用的默认值。
- `minimum`：允许的最小核大小。

输出：
- 返回一个不小于 `minimum` 的正奇数，保证可以用于 Gaussian blur、median blur、adaptive threshold 等 OpenCV 操作。

作用：
- 防止卷积核为偶数或非法值导致 OpenCV 报错。

`normalize_gray_uint8(image, name="image")`

输入：
- `image`：待处理图像，可以是灰度图、BGR 图或 BGRA 图。
- `name`：图像名称，用于错误提示。

输出：
- 单通道、`uint8`、灰度范围为 `0-255` 的图像。

作用：
- 统一完成输入检查、通道转换、浮点异常值处理和灰度归一化。
- `preprocess.py` 和 `segment_head.py` 都可以直接调用，避免重复代码。

当前完成进度：
- 已完成作业 1：`preprocess.py` 中的 ROI 预处理函数。
- 已完成作业 2：`segment_head.py` 中的四种阈值分割候选生成函数。
- 额外整理了公共工具文件 `utils.py`，用于减少重复代码。

后续工作计划：
- 在任务三中继续实现二值 mask 的形态学清理，例如开运算、闭运算、轻微腐蚀和膨胀。
- 在任务四中继续实现候选连通域筛选，从多个 mask 中选出最像精子头部的区域。
- 最终将输出的 head mask 交给 ljr，用于计算 L、W、R、SAS、LAS、HA、HD 等形态特征。

7.15：ljr：
今天主要完成精子形态筛选任务中“形态特征计算”“评分分级”“可视化复核”和“结果抽查脚本”四部分代码。我的工作重点是把 qyt 输出的精子头部二值 mask 转换成可解释的形态学指标，再根据拟合优度、长轴比和均匀度进行量化评分，最后生成 overlay 图，方便人工检查分割和评分是否合理。

目前主要编写和整理了以下 4 个 Python 文件：

1）`nk/src/sperm_morphology/features.py`

作用：
该文件负责从精子头部 `head_mask` 和预处理后的 `roi_pre` 中提取形态学特征。它是后续评分分级的基础模块，主要把二值 mask 转换成 L、W、R、SAS、LAS、HA、HD、fit_iou、gray_uniformity 和 uniformity 等指标。

核心函数：
`compute_features(head_mask, roi_pre, config)`

输入：
- `head_mask`：qyt 分割得到的精子头部二值 mask。
- `roi_pre`：qyt 预处理后的 ROI 灰度图。
- `config`：配置字典，包括显微镜标定、面积阈值和评分相关参数等。

主要处理流程：
- 使用 `find_head_contour()` 从 head mask 中寻找最大外轮廓。
- 使用 `fit_head_ellipse()` 对头部轮廓进行椭圆拟合，并保证长轴 `L` 大于等于短轴 `W`。
- 计算基础形态指标：`L_px`、`W_px`、`R`、`HA_px2`、`HD_px`。
- 如果 `config["calibration"]["um_per_pixel"]` 存在，则额外输出 `L_um`、`W_um`、`HA_um2`、`HD_um`，不伪造微米单位。
- 根据椭圆角度把 mask 旋转到主轴坐标系，计算短轴对称性 `SAS` 和长轴对称性 `LAS`。
- 生成拟合椭圆 mask，计算 head mask 和椭圆 mask 的 IoU，得到 `fit_iou`。
- 在 head mask 内统计灰度均匀度 `gray_uniformity`，并结合 `min(SAS, LAS)` 得到综合均匀度 `uniformity`。

失败处理：
- mask 为空时返回 `empty_mask`。
- 找不到轮廓时返回 `no_contour`。
- 轮廓点数小于 5 时返回 `not_enough_contour_points`。
- 椭圆拟合失败时返回 `ellipse_fit_failed`。
- 短轴过小时返回 `minor_axis_too_small`。
- mask 和 ROI 尺寸不一致时返回 `shape_mismatch`。

输出：
成功时返回包含 `success=True`、椭圆信息和全部形态指标的字典；失败时返回：

```python
{
    "success": False,
    "reason": "ellipse_fit_failed",
}
```

预期效果：
- 对每个成功 mask 都能稳定得到头部中心、长轴、短轴、角度和面积等基础形态指标。
- 可以量化头部是否接近椭圆、是否对称、内部灰度是否均匀。
- 给后续 `scoring.py` 提供稳定字段，方便 dzh 写入 CSV。

2）`nk/src/sperm_morphology/scoring.py`

作用：
该文件负责把 `features.py` 输出的连续形态指标转换成 0-100 分的子评分、总分和 A/B/C/D/Reject 等级。评分维度对应方案中的三项：拟合优度、长轴比和均匀度。

核心函数：
`score_features(features, config)`

输入：
- `features`：`compute_features()` 输出的特征字典。
- `config`：评分配置字典，支持从 `config["scoring"]` 读取权重、阈值和等级分界线。

主要评分公式：
- `fit_score = clamp((fit_iou - 0.60) / (0.85 - 0.60), 0, 1) * 100`
- `axis_score = clamp(1 - abs(R - 1.62) / 0.35, 0, 1) * 100`
- `uniformity_score = clamp((uniformity - 0.60) / (0.85 - 0.60), 0, 1) * 100`
- `total_score = 0.40 * fit_score + 0.30 * axis_score + 0.30 * uniformity_score`

硬性剔除条件：
- 特征计算失败：`Reject`。
- 头部面积 `HA_px2` 小于最小阈值：`area_too_small`。
- 头部面积 `HA_px2` 大于最大阈值：`area_too_large`。
- 长宽比 `R` 极端异常：`axis_ratio_extreme`。
- `min(SAS, LAS) < 0.55`：`low_symmetry`。
- `fit_iou < 0.50`：`bad_fit`。

输出字段：
- `fit_score`
- `axis_score`
- `uniformity_score`
- `total_score`
- `grade`
- `reject_reason`

预期效果：
- 正常样本输出 A/B/C/D 等级。
- 明显异常样本直接进入 Reject，不强行给普通等级。
- 后续调阈值时只需要改配置，不需要改评分代码。

3）`nk/src/sperm_morphology/visualize.py`

作用：
该文件负责把每个候选精子的检测、分割、拟合和评分结果画回原图，保存为 overlay 图，供人工复核。

核心函数：
`save_overlay(image, roi_info, head_mask, features, scores, config)`

输入：
- `image`：原始整图。
- `roi_info`：dzh 裁剪 ROI 时返回的坐标映射信息。
- `head_mask`：qyt 输出的头部 mask。
- `features`：ljr 计算出的形态特征。
- `scores`：ljr 计算出的评分和等级。
- `config`：配置字典，主要用于读取输出目录。

overlay 图中绘制内容：
- 原始 ROI 范围。
- 目标 bbox。
- 头部轮廓。
- 拟合椭圆。
- 椭圆长轴和短轴。
- `grade` 和 `total_score`。

输出位置：
默认保存到：

```text
outputs/overlays/
```

命名方式：
- 成功样本：`sample_target0_A_91.2.png`
- 剔除样本：`sample_target0_Reject_bad_fit.png`

预期效果：
- 人工打开 overlay 图后，可以直接看到 bbox、mask、椭圆和评分是否一致。
- 当分数异常时，可以快速判断问题来自分割、椭圆拟合还是评分阈值。
- 为后续人工抽查和调参提供直观依据。

4）`nk/scripts/review_results.py`

作用：
该脚本用于复核批处理输出的 `morphology_scores.csv`，快速统计等级分布和 Reject 原因，并随机列出每个等级的 overlay 路径供人工抽查。

运行方式：

```bash
python scripts/review_results.py --csv outputs/morphology_scores.csv --per-grade 20
```

主要功能：
- 读取 `outputs/morphology_scores.csv`。
- 统计 A/B/C/D/Reject 的数量。
- 统计 Reject 的失败原因分布。
- 按等级随机抽取若干张 overlay，打印路径给人工检查。

预期效果：
- 能快速知道当前阈值下优质、可疑和剔除样本比例。
- 能发现主要失败原因，例如 `bad_fit`、`low_symmetry` 或 `area_too_small`。
- 能帮助后续判断是需要找 qyt 调分割，还是需要调整 ljr 的评分阈值。

当前完成进度：
- 已完成作业 1：`features.py` 中的轮廓提取和椭圆拟合。
- 已完成作业 2：基础形态指标 L、W、R、HA、HD 计算。
- 已完成作业 3：SAS 和 LAS 对称性指标计算。
- 已完成作业 4：拟合优度 fit_iou、灰度均匀度 gray_uniformity 和综合均匀度 uniformity 计算。
- 已完成作业 5：统一总特征函数 `compute_features()`。
- 已完成作业 6：`scoring.py` 中的三项子评分、总分、等级和 Reject 规则。
- 已完成作业 7：明确了 CSV 中需要保留 `mask_path`、`overlay_path` 和 `reject_reason`，便于 dzh 主流程写表。
- 已完成作业 8：`visualize.py` 中的 overlay 保存函数。
- 已完成作业 9：`review_results.py` 复核脚本。

后续工作计划：
- 等 dzh 的批处理主流程完成后，把 `compute_features()`、`score_features()` 和 `save_overlay()` 接入 `batch_run.py`。
- 等真实数据跑出第一批结果后，根据人工复核结果调整 `fit_iou_good/bad`、`uniformity_good/bad`、面积阈值和长宽比目标值。
- 如果传统分割产生较多边界异常，需要和 qyt 对齐 mask 质量控制规则，避免尾部粘连或重叠杂质进入高等级样本。
