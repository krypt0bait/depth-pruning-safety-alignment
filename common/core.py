"""Calibration, layer deletion, utility evaluation and safety generation."""
import gc
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from lm_eval import simple_evaluate
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding

import config

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup():
    torch.manual_seed(config.SEED)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(config.MODEL_ID, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"   # ReplaceMe's cut selection finds the last token via the mask
    return tok


def load_model(path=config.MODEL_ID):
    kwargs = dict(dtype=torch.float16, device_map="auto", trust_remote_code=True)
    if Path(path).exists():
        kwargs["local_files_only"] = True
    return AutoModelForCausalLM.from_pretrained(path, **kwargs).eval()


def free_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str))


SLIMORCA_ROLES = {"system": "system", "human": "user", "gpt": "assistant"}


def load_calibration(tok, n_samples):
    """First n_samples SlimOrca conversations after shuffle(seed=SEED)."""
    ds = load_dataset("Open-Orca/SlimOrca", split="train")
    ds = ds.shuffle(seed=config.SEED).select(range(n_samples))

    def to_messages(example):
        return {"messages": [{"role": SLIMORCA_ROLES[t["from"]], "content": t["value"]}
                             for t in example["conversations"]]}

    def tokenize(example):
        # SlimOrca has no reasoning traces, so render without thinking mode.
        text = tok.apply_chat_template(example["messages"], tokenize=False,
                                       add_generation_prompt=False, enable_thinking=False)
        return tok(text, truncation=True, max_length=config.MAX_LEN, add_special_tokens=False)

    ds = ds.map(to_messages, remove_columns=ds.column_names)
    return ds.map(tokenize, remove_columns=ds.column_names)


def make_loader(ds, tok):
    collate = DataCollatorWithPadding(tok, padding=True, return_tensors="pt")

    def collate_to_device(batch):
        return {k: v.to(DEVICE) for k, v in collate(batch).items()}

    return DataLoader(ds.with_format("python"), batch_size=config.BATCH_SIZE,
                      shuffle=False, collate_fn=collate_to_device)


def delete_layers(model, drop):
    """Delete decoder layers in place and update the config and each layer's
    layer_idx (the KV cache is indexed by it)."""
    L = model.config.num_hidden_layers
    drop = {int(j) for j in drop}
    keep = [j for j in range(L) if j not in drop]

    model.model.layers = nn.ModuleList([model.model.layers[j] for j in keep])
    model.config.num_hidden_layers = len(keep)

    layer_types = getattr(model.config, "layer_types", None)
    if layer_types is not None and len(layer_types) == L:
        model.config.layer_types = [layer_types[j] for j in keep]
    else:
        model.config.layer_types = ["full_attention"] * len(keep)

    for new_idx, layer in enumerate(model.model.layers):
        layer.self_attn.layer_idx = new_idx
        if hasattr(layer, "attention_type"):
            layer.attention_type = model.config.layer_types[new_idx]
    return model


def save_arm(model, tok, path):
    model.save_pretrained(path)
    tok.save_pretrained(path)


def run_utility_eval(arm, model_path):
    free_gpu()
    data = simple_evaluate(model="hf", model_args=f"pretrained={model_path},dtype=float16",
                           tasks=config.TASKS, batch_size=config.LM_EVAL_BATCH)
    write_json(config.RESULTS_DIR / f"utility_{arm}_results.json", data)
    free_gpu()


STRONGREJECT_URL = ("https://raw.githubusercontent.com/alexandrasouly/strongreject/main/"
                    "strongreject_dataset/strongreject_dataset.csv")
XSTEST_URL = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
HARMBENCH_URL = ("https://raw.githubusercontent.com/centerforaisafety/HarmBench/main/"
                 "data/behavior_datasets/harmbench_behaviors_text_all.csv")


def load_strongreject():
    df = pd.read_csv(STRONGREJECT_URL)
    assert len(df) == 313
    return df


def load_xstest():
    df = pd.read_csv(XSTEST_URL)
    assert len(df) == 450 and (df["label"] == "unsafe").sum() == 200
    return df


def load_harmbench():
    """Standard behaviours only: contextual ones need a context string and
    copyright ones use a different classifier."""
    df = pd.read_csv(HARMBENCH_URL)
    assert len(df) == 400
    df = df[df["FunctionalCategory"] == "standard"].reset_index(drop=True)
    assert len(df) == 200
    return df


def load_safety_benchmarks():
    """{benchmark: (prompts, prompt_ids)}; prompt_ids match across arms."""
    sr, xs, hb = load_strongreject(), load_xstest(), load_harmbench()
    return {
        "strongreject": (sr["forbidden_prompt"].tolist(), sr.index.tolist()),
        "xstest":       (xs["prompt"].tolist(),           xs.index.tolist()),
        "harmbench":    (hb["Behavior"].tolist(),         hb.index.tolist()),
    }


@torch.no_grad()
def run_generation(model, tok, prompts):
    prev, tok.padding_side = tok.padding_side, "left"   # batched generation needs left padding
    try:
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                         add_generation_prompt=True,
                                         enable_thinking=config.ENABLE_THINKING)
                 for p in prompts]
        completions = []
        for i in range(0, len(texts), config.GEN_BATCH):
            enc = tok(texts[i:i + config.GEN_BATCH], return_tensors="pt", padding=True,
                      truncation=True, max_length=config.MAX_LEN,
                      add_special_tokens=False).to(DEVICE)
            out = model.generate(**enc, max_new_tokens=config.MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
            completions += tok.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                            skip_special_tokens=True)
        return completions
    finally:
        tok.padding_side = prev


def generate_safety(arm, model_path, tok, benchmarks):
    model = load_model(model_path)
    for benchmark, (prompts, prompt_ids) in benchmarks.items():
        completions = run_generation(model, tok, prompts)
        rows = [json.dumps({"arm": arm, "prompt_id": pid, "benchmark": benchmark,
                            "prompt": p, "completion": c})
                for pid, p, c in zip(prompt_ids, prompts, completions)]
        (config.RESULTS_DIR / f"gens_{arm}_{benchmark}.jsonl").write_text("\n".join(rows))
    del model
    free_gpu()

