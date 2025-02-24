# Liger: Linearizing Large Language Models to Gated Recurrent Structures

## Framework

<p align="center">
  <img src="assets/liger_framework.png" width="90%" />
</p>
<div align="center">
Figure 1: Liger Framework
</div>

## Environment

```bash
git clone --recurse-submodules https://github.com/OpenSparseLLMs/Linearization.git
conda create -n liger python=3.10
conda activate liger
pip install -r requirements
pip install flash-attn --no-build-isolation
```

## Linearization

1. Copy your pre-trained base model directory (e.g. Meta-Llama-3-8B) to `./checkpoints/`;
2. Modify the `config` file of the original Llama-3 base model to the `config` file of the Liger model (see `./checkpoints/liger_gla_base/config.json`);
3. Modify the linearization settings in `./configs/config.yaml ` file (e.g. liger_gla.yaml); 
4. Run the linearization script:

```bash
sh scripts/train_liger.sh
```

## Evaluation

```bash
python -m eval.harness --model hf \
    --model_args pretrained=/your/Liger/checkpoints/liger_base_model,peft=/your/Liger/checkpoints/lora_adapter_path \
    --tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
    --batch_size 64 \
    --device cuda \
    --seed 0
```

## Acknowledgements

We use the triton-implemented linear attention kernels from [fla-org/flash-linear-attention](https://github.com/fla-org/flash-linear-attention). We refer to [HazyResearch/lolcats](https://github.com/HazyResearch/lolcats) to construct our linearization training processs. The evaluation is supported by [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). Sincerely thank their contributions!

## Citation

If you find this repo useful, please cite and star our work:

```bibtex
@article{lan2025liger,
  title={Liger: Linearizing Large Language Models to Gated Recurrent Structures},
  author={Lan, Disen and Sun, Weigao and Hu, Jiaxi and Du, Jusen and Cheng, Yu},
  year={2025}
}
```