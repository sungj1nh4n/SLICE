import os
import sys
from pathlib import Path

import yaml

base_cache_dir = '/mnt/storage/LongVideoHaystack/.cache/huggingface'
with open(Path(__file__).parent / "lvbench.yaml", "r") as f:
    raw_data = f.readlines()
    safe_data = []
    for i, line in enumerate(raw_data):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data.append(line)
cache_name = yaml.safe_load("".join(safe_data))["dataset_kwargs"]["cache_dir"]


def _resolve_video(cache_dir, key):
    for ext in ("mp4", "MP4", "mkv", "webm"):
        candidate = os.path.join(cache_dir, f"{key}.{ext}")
        if os.path.exists(candidate):
            return candidate
    return os.path.join(cache_dir, f"{key}.mp4")


def lvbench_doc_to_visual(doc):
    cache_dir = os.path.join(base_cache_dir, cache_name)
    key = doc.get("key") or doc.get("video_path")
    full_path = _resolve_video(cache_dir, key) if doc.get("key") else os.path.join(cache_dir, doc["video_path"])
    if not os.path.exists(full_path):
        sys.exit(f"video path:{full_path} does not exist, please check")
    return [full_path]


def lvbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "\nAnswer the question with the option letter.")

    question = doc["question"]
    options = doc.get("options", [])
    if options:
        option_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        option_text = "\n".join(f"({option_letters[i]}) {opt}" for i, opt in enumerate(options))
        question = question + "\n" + option_text

    return pre_prompt + question + post_prompt


import re


def extract_characters_regex(s):
    s = s.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
    ]
    for prefix in answer_prefixes:
        s = s.replace(prefix, "")
    s = s.replace("(", "").replace(")", "").strip()
    if len(s.split()) > 10 and not re.search("[ABCD]", s):
        return ""
    matches = re.search(r"[ABCD]", s)
    if matches is None:
        return ""
    return matches[0]


def lvbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case videomme score), value: metric value
    """
    pred = results[0]
    pred_ans = extract_characters_regex(pred)
    # gt_ans = doc["answer"].lower().strip().replace(".", "")
    gt_ans = doc["answer"]
    score = pred_ans == gt_ans

    # return {f"videomme_perception_score": data_dict for metric in matrices}
    return {"lvbench_score": score}
