#!/bin/bash
export HF_HOME=/mnt/storage/LongVideoHaystack/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
YAML_FILE="/mnt/storage/LongVideoHaystack/lmms-eval/lmms_eval/tasks/longvideobench/longvideobench_val_v.yaml"

# ============================================================
# K=32 experiments: qwen2_vl + llava_onevision
# ============================================================
K32_FILES=(
    "include_frame_idx_bba_re_32.json"
    "include_frame_idx_bba_re_32_g1.json"
)

for json_file in "${K32_FILES[@]}"; do
    suffix="${json_file%.json}"

    # Update YAML
    sed -i "s|data_files: {\"validation\" : .*\.json}|data_files: {\"validation\" : $json_file}|" "$YAML_FILE"

    # echo "============================================"
    # echo "[K=32] Running qwen2_vl with: $json_file"
    # echo "============================================"
    # accelerate launch --num_processes 4 --main_process_port 12345 -m lmms_eval \
    #     --model qwen2_vl \
    #     --model_args pretrained=Qwen/Qwen2-VL-7B-Instruct,use_topk=True,nframes=32 \
    #     --tasks longvideobench_val_v \
    #     --batch_size 1 \
    #     --log_samples \
    #     --log_samples_suffix "qwen2_vl_7b_${suffix}" \
    #     --output_path ./results/qwen2_vl/
    # echo "Finished qwen2_vl: $json_file"
    # echo ""

    echo "============================================"
    echo "[K=32] Running llava_onevision with: $json_file"
    echo "============================================"
    accelerate launch --num_processes 4 --main_process_port 12345 -m lmms_eval \
        --model llava_onevision \
        --model_args pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,use_topk=True \
        --tasks longvideobench_val_v \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix "llava_onevision_7b_${suffix}" \
        --output_path ./results/llavaonevision/
    echo "Finished llava_onevision: $json_file"
    echo ""
done

# ============================================================
# K=64 experiments: llava_vid
# ============================================================
# K64_FILES=(
#     "include_frame_idx_bba_re_64_g0_nocritique.json"
#     "include_frame_idx_bba_re_64.json"
#     "include_frame_idx_bba_re_64_g1.json"

# )

# for json_file in "${K64_FILES[@]}"; do
#     suffix="${json_file%.json}"

#     # Update YAML
#     sed -i "s|data_files: {\"validation\" : .*\.json}|data_files: {\"validation\" : $json_file}|" "$YAML_FILE"

#     echo "============================================"
#     echo "[K=64] Running llava_vid with: $json_file"
#     echo "============================================"
#     accelerate launch --num_processes 4 --main_process_port 12345 -m lmms_eval \
#         --model llava_vid \
#         --model_args pretrained=lmms-lab/LLaVA-NeXT-Video-7B-Qwen2,conv_template=chatml_direct,video_decode_backend=decord,max_frames_num=64,overwrite=False,use_topk=True \
#         --tasks longvideobench_val_v \
#         --batch_size 1 \
#         --log_samples \
#         --log_samples_suffix "llavavid_7b_qwen_${suffix}" \
#         --output_path ./results/llavavid/
#     echo "Finished llava_vid: $json_file"
#     echo ""
# done

# echo "============================================"
# echo "All experiments completed!"
# echo "============================================"
