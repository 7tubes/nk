# qyt 作业单：ROI 预处理、精子头部分割与 mask 质量控制

## 你的任务定位

你负责把 dzh 裁剪出来的 ROI 变成“可靠的精子头部 mask”。这是整个形态筛选项目中最容易出错、也最关键的一环。后续 ljr 计算 L、W、SAS、LAS、拟合优度等指标，全部依赖你给出的头部二值图。

简单说：你做的是“图像增强 + 头部分割 + 分割失败原因判断”。

## 你需要完成的代码文件

建议你主要负责以下文件：

```text
nk/
  src/
    sperm_morphology/
      preprocess.py
      segment_head.py
  outputs/
    masks/
```

如果需要调试，也可以临时生成一些中间图，但最终中间结果要放到 `outputs/` 下面，不要散落在代码目录。

## 作业 1：实现 ROI 预处理 `preprocess.py`

你需要写一个函数：

```python
def preprocess_roi(roi_gray, config: dict):
    """
    输入 dzh 裁剪出的灰度 ROI。
    输出增强后的灰度 ROI。
    """
```

处理顺序建议为：

1. 确保输入是单通道灰度图。
2. 将灰度归一化到 `0-255`。
3. 用 Gaussian blur 或 median blur 去除噪声。
4. 做背景校正：
   - 用较大核的 Gaussian blur 得到背景图；
2   - 再把结果归一化回 `0-255`。
5. 用 CLAHE 增强局部对比度。

预期效果：

1. ROI 中精子头部边缘更清楚。
2. 背景亮度不均的问题减轻。
3. 后续阈值分割更稳定。

## 作业 2：实现阈值分割候选

在 `segment_head.py` 中实现多个候选分割方法。不要一开始只写一种，因为有些 ROI 中头部偏亮，有些偏暗。

建议实现：

```python
def threshold_candidates(roi_pre, config: dict) -> list[dict]:
    """
    返回多个二值候选结果。
    每个候选包含 mask 和方法名。
    """
```

至少尝试：

1. Otsu 正向阈值。
2. Otsu 反向阈值。
3. adaptive threshold 正向。
4. adaptive threshold 反向。

返回示例：

```python
[
    {"method": "otsu_binary", "mask": mask1},
    {"method": "otsu_binary_inv", "mask": mask2},
    {"method": "adaptive_binary", "mask": mask3},
    {"method": "adaptive_binary_inv", "mask": mask4},
]
```

预期效果：

1. 不同亮暗条件下都有机会找到头部。
2. 后续可以通过质量评分自动选出最好的 mask。

## 作业 3：实现形态学清理

你需要写：

```python
def clean_binary_mask(mask, config: dict):
    """对二值图做开运算、闭运算、腐蚀/膨胀等清理。"""
```

建议步骤：

1. 开运算：去掉孤立小噪声。
2. 闭运算：补齐头部内部小孔。
3. 轻微腐蚀：尝试断开尾部细线。
4. 必要时轻微膨胀：恢复头部边界。

注意：

1. 核不要太大，否则会把小头部直接腐蚀没。
2. 所有核大小都尽量从 `config` 读取。
3. 如果一轮腐蚀导致 mask 面积过小，要保留未腐蚀版本。

预期效果：

1. 背景小颗粒被去除。
2. 头部内部更完整。
3. 尾部细线对头部轮廓的干扰减少。

## 作业 4：实现候选连通域筛选

你需要从二值图中找所有候选区域，并选出最像精子头部的那一个。

建议实现：

```python
def extract_candidate_regions(mask, target_bbox_local, config: dict) -> list[dict]:
    """找连通域或轮廓，返回候选头部区域。"""

def score_region_candidate(region, target_bbox_local, config: dict) -> float:
    """给单个候选区域打质量分。"""
```

候选区域至少要计算：

1. 面积 `area`。
2. 外接矩形宽高。
3. 区域中心到标签框中心的距离。
4. 是否能拟合椭圆。
5. 椭圆长短轴比是否合理。
6. 轮廓点数是否不少于 5。
7. 是否过于细长，避免把尾部当头部。

推荐筛选逻辑：

1. 面积小于 `min_head_area_px` 的去掉。
2. 面积大于 `max_head_area_px` 的暂时标记为可疑。
3. 离标签框中心太远的降低分数。
4. 能拟合椭圆的优先。
5. 长宽比过大，比如大于 4，基本不是正常头部。

预期效果：

1. 当 ROI 内有多个噪声点时，可以选中真正的头部。
2. 当尾部、杂质、重叠目标干扰时，可以给出较低分或失败原因。

## 作业 5：实现主函数 `segment_head`

你最终要给 dzh 的主流程提供这个接口：

```python
def segment_head(roi_pre, target_bbox_local, config: dict):
    """
    输入：
        roi_pre: 预处理后的 ROI
        target_bbox_local: 目标框在 ROI 内的局部坐标
        config: 配置

    输出：
        head_mask: 精子头部二值 mask，失败时为 None
        quality_info: 分割质量信息和失败原因
    """
```

返回的 `quality_info` 建议包含：

```python
{
    "success": True,
    "method": "otsu_binary_inv",
    "region_score": 0.83,
    "area": 42,
    "reason": "",
}
```

失败时：

```python
{
    "success": False,
    "method": "",
    "region_score": 0,
    "area": 0,
    "reason": "no_valid_contour",
}
```

必须支持的失败原因：

| 失败原因 | 含义 |
| --- | --- |
| `no_foreground` | 阈值后没有前景 |
| `no_valid_contour` | 找不到合格轮廓 |
| `area_too_small` | 候选区域太小 |
| `area_too_large` | 候选区域太大，可能重叠或杂质 |
| `cannot_fit_ellipse` | 轮廓点不足或无法拟合椭圆 |
| `tail_connected` | 尾部和头部粘连严重 |
| `overlap_or_debris` | 可能是重叠精子或大杂质 |

预期效果：

1. 成功时输出一个只包含精子头部的二值 mask。
2. 失败时不乱输出 mask，而是给出明确失败原因。
3. 不管单个 ROI 处理是否失败，都不会影响批处理继续运行。

## 作业 6：保存 mask 调试结果

你需要支持把分割结果保存到：

```text
outputs/masks/
```

建议命名：

```text
0000_target0_mask.png
0000_target1_mask.png
```

对于失败样例，可以保存调试图：

```text
0000_target1_failed_no_valid_contour.png
```

预期效果：

1. 大家可以直接打开 mask 看分割是否合理。
2. ljr 发现特征异常时，可以回头看你的 mask。
3. 后续调参有依据，不是凭感觉改。

## 作业 7：你需要交付的结果

你最终需要提交：

1. `src/sperm_morphology/preprocess.py`
2. `src/sperm_morphology/segment_head.py`
3. 若干成功 mask 示例。
4. 若干失败 mask 或失败调试图示例。
5. 一份简短说明，写明你用了哪些分割方法、当前最常见失败原因是什么。

## 验收标准

你的任务完成后，应满足：

1. 输入一个 ROI，能输出增强后的 ROI。
2. 至少能尝试 4 种阈值候选。
3. 能对二值图做形态学清理。
4. 能选择最像头部的连通域。
5. 成功样例能输出 head mask。
6. 失败样例能返回明确失败原因。
7. 能保存 mask 到 `outputs/masks/`。

## 和其他人的接口约定

你从 dzh 接收：

```python
roi_info["roi"]
roi_info["target_bbox_local"]
config
```

你返回给 dzh：

```python
mask, quality_info
```

你给 ljr 的间接输出：

```python
head_mask
roi_pre
quality_info
```

ljr 会基于你的 `head_mask` 计算形态特征，所以你的 mask 边界越稳定，后面评分越可靠。

## 额外加分项

如果有时间，可以做：

1. 在同一个 ROI 上保存四种阈值方法的对比图。
2. 统计每种分割方法被选中的次数。
3. 统计失败原因分布。
4. 做一个简单的 `debug_segment_one.py`，输入图像编号和 target_id，只调试一个 ROI。
5. 尝试 watershed 或距离变换，处理轻微粘连情况。
