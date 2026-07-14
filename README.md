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
