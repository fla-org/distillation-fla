# Three-Stage Distillation Pipeline

This repository provides a reimplementation of the paper **"RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale"** ([arXiv:2505.03005](https://arxiv.org/abs/2505.03005)).

Our work implements the **three-stage distillation pipeline** proposed in the paper, which includes attention output alignment, logits distillation, and continued training on long sequences. This implementation is built upon the foundational codebase of the [Liger](https://github.com/OpenSparseLLMs/Linearization) project.

-----

## Environment Setup

First, clone this repository, making sure to include the submodules.

```bash
git clone --recurse-submodules https://github.com/fla-org/distillation-fla.git
cd distillation-fla

# Create and activate conda environment
conda create -n your_env_name python=3.12
conda activate your_env_name

# Install dependencies
pip install -r requirements.txt
pip install deepspeed==0.15.4
pip install flash-linear-attention
```

## Preprocess corpus

```bash
python preprocess_tokenize.py # tokenize the whole corpus
python preprocess_chunk.py --context_length 512  #for stage1 and 2
python preprocess_chunk.py --context_length 4096 # for stage3
```




## Teacher model

We expect teacher model in FLA `transformer` formats. 

## Convert DeepSpeed Checkpoint to HuggingFace Format

During training, checkpoints are saved in DeepSpeed format. Before starting the next stage or running evaluation, you need to convert these checkpoints to HuggingFace format.

Example command:
```
python convert_weight_to_hf.py --deepspeed_ckpt_path $path1 \
    --student_attn_class_name gdn_v1 \
    --hf_output_dir $path2 \
    --keep_full_attention_layers []
```




## Training: A Three-Stage Process

Our training process is divided into three distinct stages. You can run each stage using the corresponding configuration file.

### Stage 1: Attention Output Alignment

This initial stage focuses on aligning the attention outputs of the model.

```bash
deepspeed hf_train_merged.py --cfg config_rad/rapid_distill_stage1_qwen.yaml
```

After the first stage training, you need to convert the checkpoint's weight to a unified `StudentForCausalLM` model weight.
The default setting (for ): 
- Tokens: 100M 
- Training length: 512
- Peak learning rate: 1e-3
- Scheduler: Cosine
- Batch size: 96. 
- Tokens per batch: 512*96~=50K




### Stage 2: Logits Distillation

In the second stage, we perform knowledge distillation on the model's logits to transfer capabilities from a teacher model.

First, we should convert the first stage's final checkpoint to HF format (see #convert-deepspeed-checkpoint-to-huggingface-format)
 

```bash
deepspeed hf_train_merged.py --cfg config_rad/rapid_distill_stage2_qwen.yaml
```

Recommended setting:
- Tokens: 600M
- Training length: 4096



### Stage 3: Continued Training on Longer Sequences

The final stage involves continuing the training on longer sequence lengths to enhance the model's performance on extended contexts.

Again, first, we should convert Stage2's checkpoint to HF format (see #convert-deepspeed-checkpoint-to-huggingface-format)

```bash
deepspeed hf_train_merged.py --cfg config_rad/rapid_distill_stage3_qwen.yaml
```

## Evaluation

Evaluation is performed using the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). First, ensure it is installed:

```bash
cd third_party/lm-evaluation-harness
pip install -e .
```

Again again, we should convert the checkpoint to HF formats (depending on which stage you wanna evaluate, see #convert-deepspeed-checkpoint-to-huggingface-format)

Then, run the evaluation script. 

```bash
python -m eval.harness --model hf \
    --model_args pretrained="fla-hub/Qwen2.5-7B-Instruct" \
    --tasks hellaswag \
    --batch_size 16 \
    --device cuda \
    --seed 0
```





## Acknowledgements

This work is built upon the foundational [Liger](https://github.com/OpenSparseLLMs/Linearization) project. We extend our sincere gratitude to the original authors for their significant contributions.

We also use the triton-implemented linear attention kernels from [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention). We refer to [HazyResearch/lolcats](https://github.com/HazyResearch/lolcats) to construct our training process. The evaluation is supported by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Thank you for these excellent open-source efforts.

## Citation

If you use this work, please cite the original Liger paper. We also encourage you to cite this repository if it has been helpful to your research.

## Citation

If you use this work, please cite the original RADLADS paper that proposed this methodology. As our codebase is built upon Liger, we also recommend citing their work.

**Primary Method (RADLADS):**

```bibtex
@misc{goldstein2025radladsrapidattentiondistillation,
      title={RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale}, 
      author={Daniel Goldstein and Eric Alcaide and Janna Lu and Eugene Cheah},
      year={2025},
      eprint={2505.03005},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.03005}, 
}
```

**Base Codebase (Liger):**

```bibtex
@article{lan2025liger,
  title={Liger: Linearizing Large Language Models to Gated Recurrent Structures},
  author={Lan, Disen and Sun, Weigao and Hu, Jiaxi and Du, Jusen and Cheng, Yu},
  journal={arXiv preprint arXiv:2503.01496},
  year={2025}
}
```