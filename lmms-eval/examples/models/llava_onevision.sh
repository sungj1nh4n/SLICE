export HF_HOME="/mnt/storage/LongVideoHaystack/.cache/huggingface" 

export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch --num_processes=4 --main_process_port 12399 --multi_gpu -m lmms_eval \
    --model=llava_onevision \
    --model_args=pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen \
    --tasks=ai2d,chartqa,docvqa_val,mmmu_pro \
    --batch_size=1

# pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git
# pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git