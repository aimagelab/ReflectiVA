#!/bin/bash
#SBATCH --job-name=reflectiva_infoseek
#SBATCH --output=localpath/logs/train/visual_rag/release/%x-%j
#SBATCH --error=localpath/logs/train/visual_rag/release/%x-%j
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --partition=your_partition
#SBATCH --account=your_account
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-199

module load anaconda3/2022.05
module load profile/deeplrn
module load cuda/11.8

source activate reflectiva
cd localpath/reflectiva

export PYTHONPATH=.

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR="${nodelist[0]}"
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export WORLD_SIZE=$SLURM_GPUS
export LOCAL_RANK=0
export RANK=0
export PART=${SLURM_ARRAY_TASK_ID}
export TOTAL_PART=${SLURM_ARRAY_TASK_MAX}

model=aimagelab/ReflectiVA
export TOKENIZER_PATH=$model

srun --cpu_bind=none -u python rag_evaluation/infoseek/release_retrieval.py \
--model_path $model \
--model_name llava_llama_3.1 \
--index_path localpath/infoseek_EVA_text_summary \
--index_path_json localpath/infoseek_EVA_text_summary \
--entity_k 5 \
--use_eva_to_retrieve \
--retriever_path BAAI/EVA-CLIP-8B \
--retriever_processor_path openai/clip-vit-large-patch14 \
--shard_path "localpath/val-{0000..0014}.tar" \
--kb_wikipedia_path localpath/Wiki100k_ver_1_0-all.json \
--short_prompt