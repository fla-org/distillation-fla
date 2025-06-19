export CUDA_VISIBLE_DEVICES=0
export HF_HOME=/data/cl/u/yangsl66/cache/huggingface
export HF_CACHE=/data/cl/u/yangsl66/cache/huggingface
export HF_DATASETS_CACHE=/data/cl/u/yangsl66/cache/huggingface
export TRANSFORMERS_CACHE=/data/cl/u/yangsl66/cache/huggingface


python run.py --cfg configs/lolcats_at.yaml

