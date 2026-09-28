# Depth pruning and safety

ReplaceMe depth pruning of Qwen3-8B, removing n ∈ {2, 4, 9} of its 36 layers, evaluated for general utility and safety.

## Method

1. **Cut selection (Eq. 4).** On the first 256 SlimOrca calibration samples, choose the block i* that minimises the angular distance between the hidden states L_i and L_{i+n} at the last token. Blocks i*+1 … i*+n are removed.
2. **Linear transform (Eq. 6).** On all 1000 calibration samples, fit T by least squares so that block i*'s MLP output M, mapped to M·T, reproduces L_{i+n}.
3. **Fusion.** Fold T into block i*'s `down_proj` (W ← TᵀW) and delete the n blocks. No training is involved.

## Evaluation

- **Utility:** lm-eval, 0-shot, on PIQA, Winogrande, OpenBookQA and TruthfulQA-MC2.
- **Safety:** greedy generations with Qwen3 thinking mode on, for StrongREJECT (313 prompts), XSTest (450) and HarmBench (200 standard behaviours).
- **Judging:** generations were scored by gpt-oss-120B.

## Layout

```
config.py            experiment settings
run_baseline.py      unpruned model
common/core.py       calibration, layer deletion, lm-eval, benchmarks, generation
replaceme/method.py  cut selection, least-squares fit of T, fusion
replaceme/run.py     pruning and evaluation for each n
```

## Running

From this directory, after `pip install -r requirements.txt`:

```bash
python run_baseline.py
```

```bash
python -m replaceme.run
```
