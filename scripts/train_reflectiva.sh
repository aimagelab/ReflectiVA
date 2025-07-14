#!/bin/bash

cd ./ReflectiVA
source activate reflectiva

export PYTHONPATH=.
export WANDB_ENTITY=...
export WANDB_PROJECT=...
export WANDB_MODE=offline
export CINECA=1
export DEBUG=0
export MAX_RESTARTS=3

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RDZV_ID=$(date +%s)     # unique job id
# export OMP_NUM_THREADS=1

epochs=1
model_path=aimagelab/LLaVA_MORE-llama_3_1-8B-finetuning
data_path_train="<data_local_path>"
data_path_eval="<data_local_path>"
vision_tower=openai/clip-vit-large-patch14-336
export TOKENIZER_PATH=$model_path

run_name="ReflectiVA_training"
job_name="ReflectiVA_training"

torchrun \
--nnodes=4 --nproc-per-node=4 --rdzv-endpoint=$MASTER_ADDR --rdzv-id=$RDZV_ID --rdzv-backend=c10d --max_restarts=$MAX_RESTARTS \
llava/train/train_mem.py \
--deepspeed ./scripts/zero3.json \
--model_name_or_path $model_path \
--llm_backbone llama_3_1 \
--llm_pad_token pad \
--version llama_3_1 \
--seed 42 \
--visual_rag True \
--llava_instruct_multiplier 3 \
--mask_paragraph True \
--mask_answer False \
--weight_loss_special 1 \
--standard_loss True \
--template_paragraph template_rank \
--data_path_train $data_path_train \
--data_path_eval $data_path_eval \
--vision_tower $vision_tower \
--mm_projector_type mlp2x_gelu \
--mm_vision_select_layer -2 \
--mm_use_im_start_end False \
--mm_use_im_patch_token False \
--image_aspect_ratio pad \
--group_by_modality_length True \
--bf16 True \
--output_dir ./checkpoints/reflectiva/${job_name} \
--num_train_epochs $epochs \
--per_device_train_batch_size 2 \
--per_device_eval_batch_size 4 \
--gradient_accumulation_steps 4 \
--evaluation_strategy steps \
--eval_steps 100 \
--save_strategy steps \
--save_steps 100 \
--save_total_limit 600 \
--learning_rate 2e-5 \
--weight_decay 0. \
--warmup_ratio 0.03 \
--lr_scheduler_type cosine \
--logging_steps 1 \
--tf32 True \
--model_max_length 4096 \
--gradient_checkpointing True \
--dataloader_num_workers 8 \
--lazy_preprocess True \
--report_to none \
--run_name $job_name

# this training configuration has been tested on 4 nodes with 4 A100 GPUs each
# with a total number of 16 GPUs and 480GB of RAM.
