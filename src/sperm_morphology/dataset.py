import os
import yaml

# 保存数据读取过程中的异常信息
# 后续由 batch_run.py 统一生成 failed_cases.csv
FAILED_CASES = []


def load_config(config_path: str) -> dict:
    """
    读取 yaml 配置文件
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def list_image_label_pairs(config: dict) -> list:
    """
    遍历 image_dir 下的图片，并匹配同名 label 文件

    返回:
    [
        {
            "image_id": "0000",
            "image_path": ".../0000.jpg",
            "label_path": ".../0000.txt"
        }
    ]
    """
    pairs = []

    image_dir = config["data"]["image_dir"]
    label_dir = config["data"]["label_dir"]

    for filename in sorted(os.listdir(image_dir)):
        if not filename.lower().endswith(".jpg"):
            continue

        image_id = os.path.splitext(filename)[0]

        image_path = os.path.join(
            image_dir,
            filename
        )

        label_path = os.path.join(
            label_dir,
            image_id + ".txt"
        )

        if not os.path.exists(label_path):
            FAILED_CASES.append({
                "image_id": image_id,
                "reason": "missing_label"
            })
            continue

        pairs.append({
            "image_id": image_id,
            "image_path": image_path,
            "label_path": label_path
        })

    return pairs


def parse_label_file(label_path: str) -> list:
    """
    解析单个标签文件

    标签格式:
    target_id x1 y1 x2 y2

    返回:
    [
        {
            "target_id": 0,
            "bbox": [x1,y1,x2,y2]
        }
    ]
    """
    targets = []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # 空标签文件直接跳过
    if len(lines) == 0:
        return targets

    for line_number, line in enumerate(lines):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 5:
            FAILED_CASES.append({
                "label_path": label_path,
                "line": line_number,
                "reason": "bad_label_format"
            })
            continue

        try:
            target_id = int(parts[0])
            x1, y1, x2, y2 = map(float, parts[1:5])

        except ValueError:
            FAILED_CASES.append({
                "label_path": label_path,
                "line": line_number,
                "reason": "bad_coordinate"
            })
            continue

        if x2 <= x1 or y2 <= y1:
            FAILED_CASES.append({
                "label_path": label_path,
                "line": line_number,
                "reason": "invalid_bbox"
            })
            continue

        targets.append({
            "target_id": target_id,
            "bbox": [x1, y1, x2, y2]
        })

    return targets


def load_dataset(config: dict) -> list:
    """
    加载完整数据集

    返回:
    [
        {
            "image_path": "...",
            "label_path": "...",
            "image_id": "0000",
            "target_id": 0,
            "bbox": [x1,y1,x2,y2]
        }
    ]
    """
    samples = []

    pairs = list_image_label_pairs(config)

    for pair in pairs:
        targets = parse_label_file(pair["label_path"])

        for target in targets:
            samples.append({
                "image_path": pair["image_path"],
                "label_path": pair["label_path"],
                "image_id": pair["image_id"],
                "target_id": target["target_id"],
                "bbox": target["bbox"]
            })

    return samples