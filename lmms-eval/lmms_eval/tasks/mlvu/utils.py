import os
import sys
from pathlib import Path

import yaml
from loguru import logger as eval_logger

base_cache_dir = '/mnt/storage/LongVideoHaystack/.cache/huggingface'


with open(Path(__file__).parent / "mlvu_dev.yaml", "r") as f:
    raw_data_dev = f.readlines()
    safe_data_dev = []
    for i, line in enumerate(raw_data_dev):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_dev.append(line)
cache_name_dev = yaml.safe_load("".join(safe_data_dev))["dataset_kwargs"]["cache_dir"]
cache_dir_dev = os.path.join(base_cache_dir, cache_name_dev)


with open(Path(__file__).parent / "mlvu_test.yaml", "r") as f:
    raw_data_test = f.readlines()
    safe_data_test = []
    for i, line in enumerate(raw_data_test):
        # remove function definition since yaml load cannot handle it
        if "!function" not in line:
            safe_data_test.append(line)
cache_name_test = yaml.safe_load("".join(safe_data_test))["dataset_kwargs"]["cache_dir"]
cache_dir_test = os.path.join(base_cache_dir, cache_name_test)


_OPTION_LETTERS = ["A", "B", "C", "D", "E", "F"]


def _video_field(doc):
    return doc.get("video") or doc.get("video_name")


def _resolve_video_path(prefix, name):
    candidate = os.path.join(prefix, name)
    if os.path.exists(candidate):
        return candidate
    # fallback: search subdirectories one level deep (MLVU groups videos by task)
    for sub in os.listdir(prefix) if os.path.isdir(prefix) else []:
        deeper = os.path.join(prefix, sub, name)
        if os.path.exists(deeper):
            return deeper
    return candidate


def mlvu_doc_to_visual_dev(doc):
    video_path = _resolve_video_path(cache_dir_dev, _video_field(doc))
    if not os.path.exists(video_path):
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def mlvu_doc_to_visual_test(doc):
    video_path = _resolve_video_path(cache_dir_test, _video_field(doc))
    if not os.path.exists(video_path):
        sys.exit(f"video path:{video_path} does not exist, please check")
    return [video_path]


def _build_question_with_candidates(doc):
    question = doc["question"]
    candidates = doc.get("candidates")
    if candidates:
        option_text = "\n".join(f"({_OPTION_LETTERS[i]}) {c}" for i, c in enumerate(candidates))
        question = f"{question}\n{option_text}"
    return question


def mlvu_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    return pre_prompt + _build_question_with_candidates(doc) + post_prompt


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
        "Best option: (",
        "Best option: ",
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


def _answer_to_letter(doc):
    answer = doc["answer"]
    if isinstance(answer, str) and len(answer) == 1 and answer.upper() in _OPTION_LETTERS:
        return answer.upper()
    candidates = doc.get("candidates", [])
    for i, c in enumerate(candidates):
        if str(c).strip() == str(answer).strip():
            return _OPTION_LETTERS[i]
    return str(answer)


def mlvu_process_results(doc, results):
    pred = results[0]
    pred_ans = extract_characters_regex(pred)

    task_type = doc.get("task_type") or doc.get("question_type")
    gt_letter = _answer_to_letter(doc)
    data_dict = {
        "question_id": doc["question"],
        "task_type": task_type,
        "pred_answer": pred_ans,
        "answer": gt_letter,
    }
    return {"mlvu_percetion_score": data_dict}


def mlvu_aggregate_results_dev(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    TASK_TYPES = {"anomaly_reco", "count", "ego", "needle", "order", "plotQA", "topic_reasoning"}
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    task_category_scores = {}

    # Calculate and log accuracy for each task category
    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores[task_cate] = accuracy
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}%")

    # Calculate and log average accuracy across all task categories
    if TASK_TYPES:
        average_accuracy = sum(task_category_scores.values()) / len(TASK_TYPES)
    else:
        average_accuracy = 0

    eval_logger.info(f"Average Performance Across All Task Categories: {average_accuracy:.1f}%")

    return average_accuracy


def mlvu_aggregate_results_test(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    TASK_TYPES = {"anomaly_reco", "count", "ego", "needleQA", "order", "plotQA", "sportsQA", "topic_reasoning", "tutorialQA"}
    category2score = {}
    for task_type in TASK_TYPES:
        category2score[task_type] = {"correct": 0, "answered": 0}

    for result in results:
        task_type = result["task_type"]
        category2score[task_type]["answered"] += 1
        category2score[task_type]["correct"] += result["pred_answer"] == result["answer"]

    task_category_scores = {}

    # Calculate and log accuracy for each task category
    for task_cate in TASK_TYPES:
        total_correct = 0
        total_answered = 0
        for k, v in category2score.items():
            if task_cate in k:
                total_correct += v["correct"]
                total_answered += v["answered"]
        accuracy = 100 * total_correct / total_answered if total_answered > 0 else 0
        task_category_scores[task_cate] = accuracy
        eval_logger.info(f"Evaluation on Task Categories: {task_cate}: {accuracy:.1f}%")

    # Calculate and log average accuracy across all task categories
    if TASK_TYPES:
        average_accuracy = sum(task_category_scores.values()) / len(TASK_TYPES)
    else:
        average_accuracy = 0

    eval_logger.info(f"Average Performance Across All Task Categories: {average_accuracy:.1f}%")

    return average_accuracy
