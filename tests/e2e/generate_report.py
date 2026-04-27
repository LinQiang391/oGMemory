#!/usr/bin/env python3
"""Generate LoCoMo benchmark comparison report across multiple runs.

Usage:
    uv run python tests/e2e/generate_report.py --runs run1 run2 run3 run4 run5 run6 run7 run8
    uv run python tests/e2e/generate_report.py --runs run3 run5 run8
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# Use a CJK-capable font
_cjk_font = "/System/Library/Fonts/STHeiti Medium.ttc"
if os.path.exists(_cjk_font):
    fm.fontManager.addfont(_cjk_font)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_cjk_font).get_name()
    plt.rcParams["axes.unicode_minus"] = False

RESULT_DIR = Path(__file__).parent / "result"

# Category mapping
CATEGORY_NAMES = {
    "single_session": "单会话",
    "cross_session": "跨会话",
    "reasoning": "推理",
    "forgetting": "遗忘",
}


def load_run(run_name: str) -> dict[str, list[dict]]:
    """Load all judged CSVs for a run, keyed by conv-id."""
    run_dir = RESULT_DIR / run_name
    if not run_dir.exists():
        return {}

    conv_data = {}
    for f in sorted(run_dir.glob("s*_judged.csv")):
        rows = []
        with open(f, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows.append(row)
        if rows:
            conv = rows[0].get("sample_id", f.stem)
            conv_data[conv] = rows
    return conv_data


def get_accuracy(rows: list[dict]) -> tuple[int, int, float]:
    correct = sum(1 for r in rows if r.get("is_correct") == "CORRECT")
    total = len(rows)
    pct = correct / total * 100 if total else 0
    return correct, total, pct


def classify_error(row: dict) -> str:
    """Classify error type for a wrong answer."""
    response = row.get("response", "").lower()
    evidence = row.get("retrieved_evidence", "")

    if "网络" in response or "network" in response or "timed out" in response or "connection error" in response:
        return "网络错误"
    if "400" in response and ("安全" in response or "敏感" in response or "unsafe" in response.lower()):
        return "安全过滤"
    if "don't" in response or "doesn't" in response or "no information" in response or "no record" in response or "无法" in response or "没有" in response:
        if evidence and "Retrieved Memories" in evidence:
            return "有检索但不知道"
        return "无检索"
    if not evidence or "Retrieved Memories" not in evidence:
        return "无检索"
    return "答错"


def sub_classify_wrong(row: dict) -> str:
    """Sub-classify wrong answers."""
    response = row.get("response", "").lower()
    expected = row.get("expected", "").lower()

    # Check if response contains partial match
    exp_words = set(expected.split())
    resp_words = set(response.split())
    overlap = exp_words & resp_words - {"the", "a", "an", "in", "on", "at", "to", "of", "and", "is", "was", "for"}

    # Time-related keywords
    time_words = {"january", "february", "march", "april", "may", "june", "july",
                  "august", "september", "october", "november", "december",
                  "2022", "2023", "2024", "monday", "tuesday", "wednesday",
                  "thursday", "friday", "saturday", "sunday"}
    has_time_q = bool(time_words & set(response.split()))
    has_time_e = bool(time_words & set(expected.split()))

    if overlap and len(overlap) >= 2:
        return "部分正确"
    if has_time_q and has_time_e:
        return "时间错误"
    return "幻觉/其他"


def check_evidence_has_answer(row: dict) -> str:
    """Check if evidence contains the answer."""
    evidence = row.get("retrieved_evidence", "")
    expected = row.get("expected", "").strip().lower()

    if not evidence or "Retrieved Memories" not in evidence:
        return "无Evidence"

    # Simple keyword overlap check
    if not expected:
        return "不确定"

    ev_lower = evidence.lower()
    # Check for significant keyword overlap
    key_words = [w for w in expected.split() if len(w) > 3 and w not in
                 {"the", "and", "with", "from", "that", "this", "was", "were", "has", "had", "have", "been", "into", "about", "which", "their", "they", "them", "what", "when", "where", "who", "how"}]
    matches = sum(1 for w in key_words if w in ev_lower)

    if matches >= len(key_words) * 0.7 and len(key_words) >= 2:
        return "Evidence含完整答案"
    elif matches >= len(key_words) * 0.3:
        return "Evidence含部分答案"
    else:
        return "Evidence不含答案"


def plot_overall(runs_data: dict, run_names: list, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    names, accs = [], []
    for rn in run_names:
        if rn not in runs_data or not runs_data[rn]:
            continue
        all_rows = [r for rows in runs_data[rn].values() for r in rows]
        _, _, pct = get_accuracy(all_rows)
        names.append(rn)
        accs.append(pct)
    bars = ax.bar(names, accs, color=plt.cm.viridis(np.linspace(0.3, 0.9, len(names))))
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Overall Accuracy by Run")
    ax.set_ylim(60, 90)
    for bar, val in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "01_overall_accuracy.png", dpi=150)
    plt.close(fig)


def plot_per_sample(runs_data: dict, run_names: list, out_dir: Path):
    # Collect all conv-ids across all runs
    all_convs = sorted(set(
        conv for rn in run_names for conv in runs_data.get(rn, {})
    ))

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(all_convs))
    width = 0.8 / len(run_names)

    for i, rn in enumerate(run_names):
        accs = []
        for conv in all_convs:
            rows = runs_data.get(rn, {}).get(conv, [])
            _, _, pct = get_accuracy(rows) if rows else (0, 0, 0)
            accs.append(pct)
        offset = (i - len(run_names)/2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=rn)

    ax.set_xticks(x)
    ax.set_xticklabels(all_convs, rotation=45)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Sample Accuracy")
    ax.set_ylim(55, 95)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "02_per_sample_accuracy.png", dpi=150)
    plt.close(fig)


def plot_per_category(runs_data: dict, run_names: list, out_dir: Path):
    categories = ["single_session", "cross_session", "reasoning", "forgetting"]
    cat_labels = [CATEGORY_NAMES.get(c, c) for c in categories]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))
    width = 0.8 / len(run_names)

    for i, rn in enumerate(run_names):
        accs = []
        for cat in categories:
            all_rows = [r for rows in runs_data.get(rn, {}).values() for r in rows
                       if r.get("category", "").lower().replace(" ", "_") == cat]
            _, _, pct = get_accuracy(all_rows) if all_rows else (0, 0, 0)
            accs.append(pct)
        offset = (i - len(run_names)/2 + 0.5) * width
        ax.bar(x + offset, accs, width, label=rn)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_labels)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Category Accuracy")
    ax.set_ylim(50, 95)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "03_per_category_accuracy.png", dpi=150)
    plt.close(fig)


def plot_error_attribution(runs_data: dict, run_names: list, out_dir: Path):
    error_types = ["安全过滤", "网络错误", "无检索", "有检索但不知道", "答错"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(error_types))
    width = 0.8 / len(run_names)

    for i, rn in enumerate(run_names):
        all_rows = [r for rows in runs_data.get(rn, {}).values() for r in rows
                   if r.get("is_correct") != "CORRECT"]
        counts = defaultdict(int)
        for r in all_rows:
            counts[classify_error(r)] += 1
        vals = [counts.get(et, 0) for et in error_types]
        offset = (i - len(run_names)/2 + 0.5) * width
        ax.bar(x + offset, vals, width, label=rn)

    ax.set_xticks(x)
    ax.set_xticklabels(error_types, rotation=15)
    ax.set_ylabel("Count")
    ax.set_title("Error Attribution")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "04_error_attribution.png", dpi=150)
    plt.close(fig)


def plot_cross_run_delta(runs_data: dict, run_names: list, out_dir: Path):
    if len(run_names) < 2:
        return
    first, last = run_names[0], run_names[-1]
    all_convs = sorted(set(runs_data.get(first, {}).keys()) | set(runs_data.get(last, {}).keys()))

    fig, ax = plt.subplots(figsize=(14, 5))
    deltas = []
    labels = []
    for conv in all_convs:
        r1 = runs_data.get(first, {}).get(conv, [])
        r2 = runs_data.get(last, {}).get(conv, [])
        if r1 and r2:
            _, _, p1 = get_accuracy(r1)
            _, _, p2 = get_accuracy(r2)
            deltas.append(p2 - p1)
            labels.append(conv)

    colors = ["green" if d >= 0 else "red" for d in deltas]
    ax.bar(labels, deltas, color=colors, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel(f"Accuracy Delta ({last} - {first})")
    ax.set_title(f"Accuracy Change: {first} → {last}")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "07_cross_run_delta.png", dpi=150)
    plt.close(fig)


def plot_question_consistency(runs_data: dict, run_names: list, out_dir: Path):
    """Show how many questions are always correct, always wrong, or flippy."""
    # Match questions by conv-id + question text
    q_results = defaultdict(list)  # key: (conv, question) -> list of bool
    for rn in run_names:
        for conv, rows in runs_data.get(rn, {}).items():
            for r in rows:
                q = r.get("question", "").strip()
                correct = r.get("is_correct") == "CORRECT"
                q_results[(conv, q)].append(correct)

    always_correct = sum(1 for v in q_results.values() if all(v))
    always_wrong = sum(1 for v in q_results.values() if not any(v))
    flippy = sum(1 for v in q_results.values() if any(v) and not all(v))

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Always Correct", "Always Wrong", "Flippy"]
    vals = [always_correct, always_wrong, flippy]
    colors = ["#2ecc71", "#e74c3c", "#f39c12"]
    ax.pie(vals, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title(f"Question Consistency across {len(run_names)} runs")
    plt.tight_layout()
    fig.savefig(out_dir / "08_question_consistency.png", dpi=150)
    plt.close(fig)


def plot_evidence_length(runs_data: dict, run_names: list, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 5))
    data_correct = {}
    data_wrong = {}

    for rn in run_names:
        c_lens, w_lens = [], []
        for rows in runs_data.get(rn, {}).values():
            for r in rows:
                evid = r.get("retrieved_evidence", "")
                l = len(evid)
                if r.get("is_correct") == "CORRECT":
                    c_lens.append(l)
                else:
                    w_lens.append(l)
        data_correct[rn] = np.median(c_lens) if c_lens else 0
        data_wrong[rn] = np.median(w_lens) if w_lens else 0

    x = np.arange(len(run_names))
    width = 0.35
    ax.bar(x - width/2, [data_correct[rn] for rn in run_names], width, label="CORRECT", color="green", alpha=0.6)
    ax.bar(x + width/2, [data_wrong[rn] for rn in run_names], width, label="WRONG", color="red", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(run_names)
    ax.set_ylabel("Median Evidence Length (chars)")
    ax.set_title("Evidence Length: Correct vs Wrong")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "09_evidence_length.png", dpi=150)
    plt.close(fig)


def generate_report(run_names: list[str], output_dir: str | None = None):
    runs_data = {}
    for rn in run_names:
        data = load_run(rn)
        if data:
            runs_data[rn] = data
            print(f"Loaded {rn}: {len(data)} convs, {sum(len(v) for v in data.values())} questions")
        else:
            print(f"WARNING: {rn} has no data, skipping")

    active_runs = [rn for rn in run_names if rn in runs_data]
    if not active_runs:
        print("No data found!")
        sys.exit(1)

    # Create output directory
    if output_dir:
        out_dir = Path(output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = RESULT_DIR / f"report_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Generate all plots
    print("Generating plots...")
    plot_overall(runs_data, active_runs, out_dir)
    plot_per_sample(runs_data, active_runs, out_dir)
    plot_per_category(runs_data, active_runs, out_dir)
    plot_error_attribution(runs_data, active_runs, out_dir)
    plot_cross_run_delta(runs_data, active_runs, out_dir)
    plot_question_consistency(runs_data, active_runs, out_dir)
    plot_evidence_length(runs_data, active_runs, out_dir)

    # Generate markdown report
    print("Generating report...")
    md = []
    md.append("# LoCoMo Benchmark 分析报告\n")
    md.append(f"Runs: {', '.join(active_runs)}\n")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # 1. Overall
    md.append("## 1. 总览\n")
    md.append("| Run | 总题数 | 正确 | 准确率 |")
    md.append("|-----|--------|------|--------|")
    for rn in active_runs:
        all_rows = [r for rows in runs_data[rn].values() for r in rows]
        c, t, pct = get_accuracy(all_rows)
        md.append(f"| {rn} | {t} | {c} | **{pct:.1f}%** |")
    md.append("")
    md.append("![总体准确率](./01_overall_accuracy.png)\n")

    # 2. Per sample
    md.append("## 2. 按 Sample 分析\n")
    all_convs = sorted(set(conv for rn in active_runs for conv in runs_data[rn]))

    header = "| Conv | " + " | ".join(active_runs) + " |"
    sep = "|------|" + "|".join(["--------" for _ in active_runs]) + "|"
    md.append(header)
    md.append(sep)
    for conv in all_convs:
        vals = []
        for rn in active_runs:
            rows = runs_data[rn].get(conv, [])
            if rows:
                _, _, pct = get_accuracy(rows)
                vals.append(f"{pct:.1f}%")
            else:
                vals.append("-")
        md.append(f"| {conv} | " + " | ".join(vals) + " |")
    md.append("")
    md.append("![按Sample准确率](./02_per_sample_accuracy.png)\n")

    # 3. Per category
    md.append("## 3. 按类别分析\n")
    categories = ["single_session", "cross_session", "reasoning", "forgetting"]
    cat_labels = {c: CATEGORY_NAMES.get(c, c) for c in categories}

    header = "| 类别 | " + " | ".join(active_runs) + " |"
    sep = "|------|" + "|".join(["--------" for _ in active_runs]) + "|"
    md.append(header)
    md.append(sep)
    for cat in categories:
        vals = []
        for rn in active_runs:
            cat_rows = [r for rows in runs_data[rn].values() for r in rows
                       if r.get("category", "").lower().replace(" ", "_") == cat]
            if cat_rows:
                _, _, pct = get_accuracy(cat_rows)
                vals.append(f"{pct:.1f}%")
            else:
                vals.append("-")
        md.append(f"| {cat_labels[cat]} | " + " | ".join(vals) + " |")
    md.append("")
    md.append("![按类别准确率](./03_per_category_accuracy.png)\n")

    # 4. Error attribution per run
    md.append("## 4. 错误归因\n")
    for rn in active_runs:
        wrong_rows = [r for rows in runs_data[rn].values() for r in rows
                     if r.get("is_correct") != "CORRECT"]
        total_errors = len(wrong_rows)
        md.append(f"### {rn}\n")
        md.append(f"总错误: {total_errors}\n")

        error_counts = defaultdict(int)
        for r in wrong_rows:
            error_counts[classify_error(r)] += 1

        md.append("| 错误类型 | 数量 | 占比 |")
        md.append("|----------|------|------|")
        for et in ["无检索", "网络错误", "安全过滤", "有检索但不知道", "答错"]:
            cnt = error_counts.get(et, 0)
            pct = cnt / total_errors * 100 if total_errors else 0
            md.append(f"| {et} | {cnt} | {pct:.1f}% |")
        md.append("")

        # Sub-classify "答错"
        wrong_answer_rows = [r for r in wrong_rows if classify_error(r) == "答错"]
        if wrong_answer_rows:
            sub_counts = defaultdict(int)
            for r in wrong_answer_rows:
                sub_counts[sub_classify_wrong(r)] += 1
            md.append("#### 答错细分\n")
            md.append("| 子类型 | 数量 | 占比 |")
            md.append("|--------|------|------|")
            for st in ["部分正确", "时间错误", "幻觉/其他"]:
                cnt = sub_counts.get(st, 0)
                pct = cnt / len(wrong_answer_rows) * 100 if wrong_answer_rows else 0
                md.append(f"| {st} | {cnt} | {pct:.1f}% |")
            md.append("")

        # Evidence analysis for "有检索但不知道"
        dk_rows = [r for r in wrong_rows if classify_error(r) == "有检索但不知道"]
        if dk_rows:
            ev_counts = defaultdict(int)
            for r in dk_rows:
                ev_counts[check_evidence_has_answer(r)] += 1
            md.append(f'#### "不知道"题目的 Evidence 分析\n')
            md.append("| Evidence 含答案情况 | 数量 | 占比 |")
            md.append("|-------------------|------|------|")
            for et in ["Evidence含完整答案", "Evidence含部分答案", "Evidence不含答案", "无Evidence", "不确定"]:
                cnt = ev_counts.get(et, 0)
                pct = cnt / len(dk_rows) * 100 if dk_rows else 0
                md.append(f"| {et} | {cnt} | {pct:.1f}% |")
            md.append("")

        # Error by category
        md.append("#### 按 Category 错误分布\n")
        cat_header = "| 错误类型 | " + " | ".join(cat_labels.values()) + " |"
        cat_sep = "|----------|" + "|".join(["---|" for _ in categories])
        md.append(cat_header)
        md.append(cat_sep)
        for et in ["无检索", "网络错误", "安全过滤", "有检索但不知道", "答错"]:
            vals = []
            for cat in categories:
                cat_rows = [r for r in wrong_rows if r.get("category", "").lower().replace(" ", "_") == cat]
                cnt = sum(1 for r in cat_rows if classify_error(r) == et)
                vals.append(str(cnt))
            md.append(f"| {et} | " + " | ".join(vals) + " |")
        md.append("")

    md.append("![错误归因](./04_error_attribution.png)\n")

    # 5. Token usage
    md.append("## 5. Token 消耗\n")
    for rn in active_runs:
        all_rows = [r for rows in runs_data[rn].values() for r in rows]
        total_in = sum(int(r.get("input_tokens", 0)) for r in all_rows)
        total_out = sum(int(r.get("output_tokens", 0)) for r in all_rows)
        total_cache = sum(int(r.get("cacheRead", 0)) for r in all_rows)
        total_tok = sum(int(r.get("total_tokens", 0)) for r in all_rows)
        md.append(f"### {rn}\n")
        md.append(f"- Input: {total_in:,}")
        md.append(f"- Output: {total_out:,}")
        md.append(f"- Cache Read: {total_cache:,}")
        md.append(f"- Total: {total_tok:,}\n")

    # 6. Cross-run delta
    md.append("## 6. 跨 Run 波动\n")
    if len(active_runs) >= 2:
        first, last = active_runs[0], active_runs[-1]
        # Match by (conv, question)
        first_map = {}
        for conv, rows in runs_data[first].items():
            for r in rows:
                first_map[(conv, r.get("question","").strip())] = r.get("is_correct") == "CORRECT"

        always_correct = always_wrong = flippy = 0
        for conv, rows in runs_data[last].items():
            for r in rows:
                key = (conv, r.get("question","").strip())
                if key in first_map:
                    last_correct = r.get("is_correct") == "CORRECT"
                    if first_map[key] and last_correct:
                        always_correct += 1
                    elif not first_map[key] and not last_correct:
                        always_wrong += 1
                    else:
                        flippy += 1

        total = always_correct + always_wrong + flippy
        md.append(f"对比: {first} → {last}\n")
        md.append(f"- 始终正确: {always_correct} ({always_correct/total*100:.1f}%)")
        md.append(f"- 始终错误: {always_wrong} ({always_wrong/total*100:.1f}%)")
        md.append(f"- 翻转: {flippy} ({flippy/total*100:.1f}%)\n")

    md.append("![准确率变化](./07_cross_run_delta.png)\n")
    md.append("![题目一致性](./08_question_consistency.png)\n")

    # 7. Evidence quality
    md.append("## 7. 检索质量\n")
    for rn in active_runs:
        all_rows = [r for rows in runs_data[rn].values() for r in rows]
        no_evid = sum(1 for r in all_rows if "Retrieved Memories" not in r.get("retrieved_evidence", ""))
        c_lens = [len(r.get("retrieved_evidence","")) for r in all_rows if r.get("is_correct") == "CORRECT"]
        w_lens = [len(r.get("retrieved_evidence","")) for r in all_rows if r.get("is_correct") != "CORRECT"]
        md.append(f"### {rn}\n")
        md.append(f"- 无 Evidence 率: {no_evid/len(all_rows)*100:.1f}%")
        md.append(f"  - CORRECT: median evidence = {np.median(c_lens):.0f} chars")
        md.append(f"  - WRONG: median evidence = {np.median(w_lens):.0f} chars\n")

    md.append("![Evidence长度](./09_evidence_length.png)\n")

    # 8. Data source
    md.append("## 8. 数据源\n")
    for rn in active_runs:
        md.append(f"- `{RESULT_DIR / rn}/`")

    # Write report
    report_path = out_dir / "report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(f"\nReport generated: {report_path}")
    return str(out_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate LoCoMo benchmark comparison report")
    parser.add_argument("--runs", nargs="+", required=True, help="Run names to compare (e.g. run1 run2 run3)")
    parser.add_argument("--output", "-o", help="Output directory (default: auto-generated)")
    args = parser.parse_args()

    generate_report(args.runs, args.output)


if __name__ == "__main__":
    main()
