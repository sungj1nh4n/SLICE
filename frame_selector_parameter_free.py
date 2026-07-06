import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import sys
import os
import json
import multiprocessing
from functools import partial
from tqdm import tqdm
import math

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

# =========================================================
# File I/O Helpers (기존 유지)
# =========================================================

def load_lvb_entries(json_path: str):
    """
    LongVideoBench JSON 파일에서 엔트리 리스트를 로드합니다.
    """
    def _read_json_robust(path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {path}")
        file_size = os.path.getsize(path)
        if file_size == 0:
            raise ValueError(f"입력 파일이 비어 있습니다: {path}")

        last_decode_error = None
        for enc in ["utf-8", "utf-8-sig", "latin-1"]:
            try:
                with open(path, "r", encoding=enc) as f:
                    text = f.read()
                break
            except UnicodeDecodeError as e:
                last_decode_error = e
                text = None
        else:
            try:
                import chardet
                with open(path, "rb") as fb:
                    raw = fb.read()
                detected = chardet.detect(raw)
                enc = detected.get("encoding")
                if not enc:
                    raise UnicodeDecodeError("unknown", b"", 0, 1, "encoding detection failed")
                text = raw.decode(enc, errors="strict")
            except Exception as e:
                raise UnicodeDecodeError(
                    "unknown", b"", 0, 1,
                    f"Failed to decode file. Last error: {last_decode_error or e}"
                )

        stripped = text.strip()
        if not stripped:
            raise ValueError(f"입력 파일이 비어 있거나 공백만 포함합니다: {path}")

        if stripped[0] == '[':
            return json.loads(stripped)

        if stripped[0] == '{':
            obj = json.loads(stripped)
            return [obj]

        entries = []
        for ln in stripped.splitlines():
            s = ln.strip()
            if not s: continue
            if s.startswith('//'): continue
            try:
                entries.append(json.loads(s))
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"지원하지 않는 JSON 형식이거나 손상된 파일일 수 있습니다: {path}", s, e.pos
                ) from e
        if entries:
            return entries

        raise json.JSONDecodeError(f"JSON/JSONL 형식이 아닙니다: {path}", stripped[:50], 0)

    return _read_json_robust(json_path)

def get_video_id_from_entry(entry: dict) -> str:
    """Video ID 우선순위: videoID > video_id > videoId > video > vid > id"""
    for key in ["videoID", "video_id", "videoId", "video", "vid", "id"]:
        val = entry.get(key)
        if val is not None and str(val) != "":
            return str(val)
    return ""

def _to_jsonable(obj):
    """np.ndarray/np.generic/pandas types를 JSON 직렬화 가능 형태로 변환 (재귀)."""
    if isinstance(obj, dict):
        return {_to_jsonable(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    if isinstance(obj, np.ndarray):
        return _to_jsonable(obj.tolist())
    if PANDAS_AVAILABLE:
        pd = __import__('pandas')
        if hasattr(obj, 'to_pydatetime'):
            return str(obj)
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj

# =========================================================
# New Sampling Class: Parameter-Free Iso-Semantic
# =========================================================

class ParameterFreeSampler:
    """
    Parameter-Free Iso-Semantic Sampler (EBE)
    Core Logic: Warps the physical timeline into a semantic timeline 
                to achieve equal information spacing.
    """
    
    def __init__(self):
        pass

    def sample_iso_semantic(self, frame_indices, scores, k):
        """
        Executes Iso-Semantic Sampling.
        
        Args:
            frame_indices (list): List of original frame indices.
            scores (np.array): Raw relevance scores (BLIP/CLIP scores).
            k (int): Number of frames to select.
            
        Returns:
            selected_indices (list): List of selected frame indices (sorted).
            debug_info (dict): Data for visualization (smoothed_scores, cdf, boundaries).
        """
        n = len(scores)
        scores = np.array(scores, dtype=np.float64)

        # -------------------------------------------------------------
        # 1. Scale-Adaptive Smoothing (Log-Scale Sigma)
        # Novelty: Adapts to video length automatically.
        # -------------------------------------------------------------
        adaptive_sigma = np.log(n) if n > 1 else 1.0
        smooth_scores = gaussian_filter(scores, sigma=adaptive_sigma)

        # -------------------------------------------------------------
        # 2. Semantic Density (PDF) & Timeline (CDF) Construction
        # -------------------------------------------------------------
        # 음수 값 제거 및 정규화
        pdf = np.maximum(smooth_scores, 0)
        total_energy = np.sum(pdf)
        
        # Prepare valid debug info even for short videos
        if total_energy < 1e-9:
            # Fallback for zero energy
            cdf = np.linspace(0, 1, n)
        else:
            pdf_norm = pdf / total_energy
            cdf = np.cumsum(pdf_norm)
            cdf[-1] = 1.0

        # 예외 처리: 요청 프레임 수가 전체 길이보다 많거나 같으면 전체 반환
        if n <= k:
            debug_info = {
                'smooth_scores': smooth_scores,
                'cdf': cdf,
                'time_boundaries': [] # No partitioning needed
            }
            return sorted(frame_indices), debug_info

        # 모든 점수가 0인 경우 (예외) -> Uniform Sampling
        if total_energy < 1e-9:
            indices = np.linspace(0, n - 1, k, dtype=int)
            return sorted([frame_indices[i] for i in indices]), None

        # -------------------------------------------------------------
        # 3. Iso-Semantic Partitioning
        # Novelty: Divide 'Energy' axis, not 'Time' axis
        # -------------------------------------------------------------
        # 에너지 축(y축)을 K등분
        energy_boundaries = np.linspace(0, 1, k + 1)
        
        # 역함수(Inverse CDF)를 통해 시간 축(x축)의 경계선 찾기 (Warping)
        time_boundaries = np.searchsorted(cdf, energy_boundaries)
        
        selected_indices = []
        
        # -------------------------------------------------------------
        # 4. Local Maximization in Warped Windows
        # -------------------------------------------------------------
        for i in range(k):
            start_idx = time_boundaries[i]
            end_idx = time_boundaries[i+1]
            
            # 인덱스 보정: 구간이 너무 좁거나(start==end) 역전된 경우 방지
            if start_idx >= end_idx:
                end_idx = min(start_idx + 1, n)
                start_idx = max(0, end_idx - 1)
            
            # 해당 윈도우 내에서 가장 선명한 정보(Raw Score Peak)를 선택
            # 스무딩된 점수는 '구간'을 정하는 데 쓰고, 실제 선택은 '원본'을 사용
            window_scores = scores[start_idx:end_idx]
            
            if len(window_scores) > 0:
                local_max_offset = np.argmax(window_scores)
                global_idx = start_idx + local_max_offset
                selected_indices.append(frame_indices[min(global_idx, n-1)])
            else:
                fallback = min(start_idx, n-1)
                selected_indices.append(frame_indices[fallback])

        # 중복 제거 및 정렬
        final_selection = sorted(list(set(selected_indices)))
        
        # 시각화를 위한 정보 패키징
        debug_info = {
            'smooth_scores': smooth_scores,
            'cdf': cdf,
            'time_boundaries': time_boundaries
        }
        
        return final_selection, debug_info


# =========================================================
# Visualization Helper (Updated for EBE/CDF)
# =========================================================
def save_visualization(entry, raw_scores, output_dir, frame_indices, selected_indices, debug_info):
    try:
        os.makedirs(output_dir, exist_ok=True)
        video_id = get_video_id_from_entry(entry)
        question_id = entry.get('id', str(entry.get('question_idx', str(entry.get('video_idx', 'unknown')))))
        question = entry.get('question', '')
        
        # EBE Debug Data
        smooth_scores = debug_info['smooth_scores']
        cdf = debug_info['cdf']
        time_boundaries = debug_info['time_boundaries']
        
        fig, ax1 = plt.subplots(figsize=(16, 6))

        # [Left Axis] Score Plot
        ax1.set_xlabel('Frame Index')
        ax1.set_ylabel('Relevance Score', color='tab:blue')
        ax1.plot(raw_scores, color='tab:blue', alpha=0.3, label='Raw BLIP Score')
        ax1.plot(smooth_scores, color='navy', linewidth=2, label=f'Semantic Density (sigma={np.log(len(raw_scores)):.1f})')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Y축 범위 설정 (그래프 여백 확보)
        max_val = max(np.max(raw_scores) if len(raw_scores)>0 else 1.0, np.max(smooth_scores) if len(smooth_scores)>0 else 1.0)
        ax1.set_ylim(0, max_val * 1.15)

        # [Right Axis] CDF Plot (Novelty Visualization)
        ax2 = ax1.twinx() 
        ax2.set_ylabel('Cumulative Energy (CDF)', color='tab:orange')
        ax2.plot(cdf, color='tab:orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Energy CDF')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.set_ylim(0, 1.05)
        
        # [Partition Boundaries]
        # 구간 경계선 그리기 (초록색 점선)
        for tb in time_boundaries:
            ax1.axvline(x=tb, color='green', linestyle=':', alpha=0.5)

        # [Selected Frames]
        # frame_indices 값을 index로 매핑
        val_to_idx = {val: i for i, val in enumerate(frame_indices)}
        plotted_indices = [val_to_idx[v] for v in selected_indices if v in val_to_idx]
        
        ax1.scatter(plotted_indices, [raw_scores[i] for i in plotted_indices], 
                    color='red', marker='*', s=150, zorder=10, label='Iso-Semantic Selected')

        # === New: Visualize GT Position ===
        positions = entry.get('position', [])
        if positions:
            gt_indices = []
            for pos in positions:
                try:
                    if pos in frame_indices:
                        idx = frame_indices.index(pos)
                        gt_indices.append(idx)
                    else:
                        arr = np.array(frame_indices)
                        idx = (np.abs(arr - pos)).argmin()
                        gt_indices.append(idx)
                except:
                    pass
            
            if gt_indices:
                ax1.scatter(gt_indices, [raw_scores[i] for i in gt_indices], 
                            color='red', marker='x', s=100, zorder=11, label='GT Position')
                for idx in gt_indices:
                    ax1.axvline(x=idx, color='red', linewidth=1.5, linestyle='--', alpha=0.8)

        q_text = question[:80] + "..." if len(question) > 80 else question
        plt.title(f"Iso-Semantic Sampling (Parameter-Free) | Q: {q_text}", fontsize=12, fontweight='bold')
        
        # Legend 통합
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{question_id}.png")
        plt.savefig(save_path, dpi=100)
        plt.close()
        
    except Exception as e:
        print(f"Error visualizing {entry.get('question_idx')}: {e}")
        import traceback
        traceback.print_exc()

# =========================================================
# Worker Function for Multiprocessing
# =========================================================
def process_single_entry(entry_data, k, round_to_int, sort_by_score=False, visualize=False, visualization_dir=None):
    """
    Worker function that runs in a separate process.
    entry_data: (entry, blip_scores, frame_indices)
    """
    entry, blip_scores, frame_indices = entry_data
    
    try:
        # Validate data integrity
        n_scores = len(blip_scores)
        n_frames = len(frame_indices)
        
        if n_scores != n_frames:
            n_min = min(n_scores, n_frames)
            blip_scores = blip_scores[:n_min]
            frame_indices = frame_indices[:n_min]
        
        raw_clip_scores = np.array(blip_scores)
        # Normalize scores to 0-1
        raw_clip_scores = np.clip(raw_clip_scores, 0, 1)
        
        # -----------------------------------------------------
        # [NEW] Execute Parameter-Free Sampler
        # -----------------------------------------------------
        sampler = ParameterFreeSampler()
        
        # No kappa, lambda, alpha needed. Just K.
        selected_indices, debug_info = sampler.sample_iso_semantic(
            frame_indices, 
            raw_clip_scores, 
            k=k
        )
        # -----------------------------------------------------

        if sort_by_score:
            # Sort by score ascending (larger score at the back) - Optional
            val_to_idx = {val: i for i, val in enumerate(frame_indices)}
            selected_indices = sorted(selected_indices, key=lambda x: raw_clip_scores[val_to_idx[x]])
        else:
            # Sort by temporal order (index) - Default behavior
            selected_indices = sorted(selected_indices)
        
        entry["frame_idx"] = selected_indices
        entry["frame_num"] = len(selected_indices)
        entry["method"] = "iso_semantic_ebe" # 기록용 태그 변경

        if visualize and visualization_dir and debug_info:
            save_visualization(entry, raw_clip_scores, visualization_dir, frame_indices, selected_indices, debug_info)
        
    except Exception as e:
        print(f"Error processing {entry.get('video_idx', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        entry["frame_idx"] = []
        
    return entry


# =========================================================
# Optimized Main Processor (기존 유지)
# =========================================================
def process_lvb_dataset_parallel(
    input_json_path="/mnt/storage/LongVideoHaystack/.cache/huggingface/longvideobench/lvb_val.json",
    output_json_path="/mnt/storage/LongVideoHaystack/include_frame_idx.json",
    blip_score_path="/mnt/storage/LongVideoHaystack/AKS/outscores/longvideobench/blip/scores.json",
    frames_json_path="/mnt/storage/LongVideoHaystack/AKS/outscores/longvideobench/blip/frames.json",
    k=32,
    round_to_int=True,
    sort_by_score=False,
    max_entries=None,
    num_workers=16,
    visualize=False,
    visualization_dir="/mnt/storage/LongVideoHaystack/visualize"
):
    """
    Main processor using precomputed scores and frame indices.
    """
    # 1. Load Data
    print(f"📂 Loading entries from {input_json_path}...")
    entries = load_lvb_entries(input_json_path)
    if max_entries:
        entries = entries[:max_entries]

    # 2. Load Precomputed BLIP Scores and Frames
    print(f"⚡ Loading precomputed BLIP scores from {blip_score_path}...")
    with open(blip_score_path, "r") as f:
        all_blip_scores = json.load(f)
        
    print(f"🎞️ Loading precomputed Frame Indices from {frames_json_path}...")
    with open(frames_json_path, "r") as f:
        all_frame_indices = json.load(f)
        
    # Validation
    if len(all_blip_scores) != len(entries):
        print(f"⚠️ Warning: Mismatch in entries ({len(entries)}) vs scores ({len(all_blip_scores)}).")
    
    if len(all_blip_scores) != len(all_frame_indices):
        print(f"⚠️ Critical Warning: Scores count ({len(all_blip_scores)}) != Frames count ({len(all_frame_indices)}). Please check data integrity.")

    # 4. Prepare Args
    worker_args = []
    # Zip together entries, scores, and frames
    for i, entry in enumerate(entries):
        if i < len(all_blip_scores) and i < len(all_frame_indices):
            worker_args.append((entry, all_blip_scores[i], all_frame_indices[i]))
        
    # 5. Parallel Processing
    print(f"🔥 Processing {len(worker_args)} items using {num_workers} workers...")
    
    worker_fn = partial(
        process_single_entry,
        k=k,
        round_to_int=round_to_int,
        sort_by_score=sort_by_score,
        visualize=visualize,
        visualization_dir=visualization_dir
    )
    
    augmented_entries = []
    
    with multiprocessing.Pool(processes=num_workers) as pool:
        for result in tqdm(pool.imap(worker_fn, worker_args), total=len(worker_args), desc="Iso-Semantic Sampling"):
            augmented_entries.append(result)
            
    # 6. Save
    print(f"💾 Saving results to {output_json_path}...")
    safe_augmented = [_to_jsonable(e) for e in augmented_entries]
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(safe_augmented, f, ensure_ascii=False)
        
    print(f"✅ Done! Output saved to {output_json_path}")
    return output_json_path


if __name__ == "__main__":
    from datetime import datetime
    timestamp = datetime.now().strftime("%m%d%H%M")
    
    # -------------------------------------------------------------
    # 사용자 설정 경로 (필요시 수정)
    # -------------------------------------------------------------
    output_path = f"/mnt/storage/LongVideoHaystack/include_frame_idx_IsoSemantic_{timestamp}.json"
    vis_dir = f"/mnt/storage/LongVideoHaystack/visualizations/iso_semantic_{timestamp}"

    input_entries = "/mnt/storage/LongVideoHaystack/.cache/huggingface/videomme/videomme.json"
    scores_path = "/mnt/storage/LongVideoHaystack/AKS/outscores/videomme/blip/scores.json"
    frames_path = "/mnt/storage/LongVideoHaystack/AKS/outscores/videomme/blip/frames.json"
    
    # Run
    process_lvb_dataset_parallel(
        input_json_path=input_entries,
        output_json_path=output_path,
        blip_score_path=scores_path,
        frames_json_path=frames_path,
        k=32,  # Iso-Semantic Strategy는 K만 입력받습니다.
        max_entries=None,
        num_workers=min(multiprocessing.cpu_count(), 32),
        visualize=False, # 시각화 켜기 (Novelty 확인용)
        visualization_dir=vis_dir
    )