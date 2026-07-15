import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def print_counter(title, counter):
    print(title)
    if not counter:
        print("  无")
        return
    for key, count in counter.most_common():
        print(f"  {key or '空'}: {count}")


def sample_overlays(rows, per_grade, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("grade", "")].append(row.get("overlay_path", ""))

    random.seed(seed)
    for grade in sorted(grouped):
        overlays = [path for path in grouped[grade] if path]
        if not overlays:
            continue
        picks = random.sample(overlays, min(per_grade, len(overlays)))
        print(f"\n{grade} 抽查 overlay:")
        for path in picks:
            print(f"  {path}")


def main():
    parser = argparse.ArgumentParser(description="复核 morphology_scores.csv 的等级和失败原因。")
    parser.add_argument("--csv", default="outputs/morphology_scores.csv", help="结果 CSV 路径")
    parser.add_argument("--per-grade", type=int, default=20, help="每个等级随机抽查数量")
    parser.add_argument("--seed", type=int, default=20260715, help="随机种子")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"找不到 CSV 文件：{csv_path}")

    rows = read_rows(csv_path)
    grade_counter = Counter(row.get("grade", "") for row in rows)
    reject_counter = Counter(
        row.get("reject_reason", "")
        for row in rows
        if row.get("grade", "") == "Reject"
    )

    print(f"总记录数: {len(rows)}")
    print_counter("等级分布:", grade_counter)
    print_counter("Reject 原因分布:", reject_counter)
    sample_overlays(rows, max(args.per_grade, 0), args.seed)


if __name__ == "__main__":
    main()
