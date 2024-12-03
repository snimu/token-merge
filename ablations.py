# /// script
# requires-python = "==3.12"
# dependencies = [
#   "torch",
#   "transformers",
#   "polars",
#   "tqdm",
# ]
# ///

"""
INSTRUCTIONS:

- you need to run `huggingface-cli login` once and add your token
- you need access to the meta-llama weights

ON LAMBDA LABS:

- you need to run `pip install -U torch torchvision` before running this script
"""

import argparse
import os
from typing import Literal
import itertools

import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import polars as pl
from tqdm import tqdm


RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'

PYTHIA_MODELS = {
    "pythia-14m": "EleutherAI/pythia-14m",
    "pythia-70m": "EleutherAI/pythia-70m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "pythia-410m": "EleutherAI/pythia-410m",
    "pythia-1b": "EleutherAI/pythia-1b",
    "pythia-1.4b": "EleutherAI/pythia-1.4b",
    "pythia-2.8b": "EleutherAI/pythia-2.8b",
    "pythia-6.9b": "EleutherAI/pythia-6.9b",
    "pythia-12b": "EleutherAI/pythia-12b",
}
PYTHIA_MODELS_TYPE = Literal[
    "pythia-14m", "pythia-70m", 
    "pythia-160m", "pythia-410m", 
    "pythia-1b", "pythia-1.4b", 
    "pythia-2.8b", "pythia-6.9b", 
    "pythia-12b"
]


def load_model_and_tokenizer(
        model_name: PYTHIA_MODELS_TYPE,
        deduped: bool = False,
) -> tuple[GPTNeoXForCausalLM, AutoTokenizer]:
    download_name = PYTHIA_MODELS[model_name] + ("-deduped" if deduped else "")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(download_name)
    model = GPTNeoXForCausalLM.from_pretrained(download_name).to(device) 
    return model, tokenizer


def load_and_tokenize_wikitext(
        tokenizer: AutoTokenizer,
        shuffle: bool = False, 
        seed: int | None = None, 
        n_texts: int = 128,
) -> list[dict[str, torch.Tensor]]:
    df = pl.read_parquet(
        'hf://datasets/Salesforce/wikitext'
        '/wikitext-103-v1/test-00000-of-00001.parquet'
    )
    if shuffle:
        df = df.sample(n=len(df), seed=seed)
    texts = [text for text in df["text"].to_list() if len(text) > 128*5][:n_texts]
    dataset = [tokenizer(text, return_tensors="pt") for text in texts]
    assert all(sample.input_ids.shape[-1] != 0 for sample in dataset)
    return dataset


def measure_compression(
        model_name: PYTHIA_MODELS_TYPE,
        model: GPTNeoXForCausalLM, 
        tokenizer: AutoTokenizer, 
        dataset: list[dict[str, torch.Tensor]],
        savefile: str,
        max_gen_tokens: int = 1024,
        window_size: int = 64,
        temperature: float = 0.0,
        loop: tqdm = None,
        device: str | torch.device = "cuda",
) -> None:
    assert max_gen_tokens % window_size == 0
    if temperature > 0.0:
        model.generation_config.temperature = temperature
    model.generation_config.max_new_tokens = model.generation_config.min_new_tokens = max_gen_tokens
    model.generation_config.do_sample = temperature > 0.0
    model.generation_config.pad_token_id = tokenizer.eos_token_id

    for idx, inputs in enumerate(dataset):
        loop.set_description(
            f"model={model_name}, temperature={temperature} "
            f"({idx+1}/{len(dataset)})"
        ) 
        input_ids = inputs["input_ids"].to(device)
        n_tokens_in = input_ids.shape[-1]
        assert n_tokens_in > 0, f"{input_ids.shape=}"
        attention_mask = inputs["attention_mask"].to(device)

        output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask)
        output_ids = output_ids[:, n_tokens_in:]
        n_tokens_out = output_ids.shape[-1]

        # Stats for entire output
        
        output_ids_compressed = tokenizer(
            tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0], 
            return_tensors="pt"
        ).input_ids
        n_tokens_out_reduced = output_ids_compressed.shape[-1]
        compression_ratio = (n_tokens_out - n_tokens_out_reduced) / n_tokens_out

        # Stats for moving window (no overlap)
        n_tokens_out_reduced_window_list = []
        compression_ratios_window_list = []
        for i in range(0, n_tokens_out, window_size):
            window_ids = output_ids[:, i : i + window_size if i + window_size < n_tokens_out else None]
            window_text = tokenizer.batch_decode(window_ids, skip_special_tokens=True)
            if not window_text:
                break
            window_text = window_text[0]
            n_tokens_out_reduced_window = tokenizer(window_text, return_tensors="pt").input_ids.shape[-1]
            compression_ratio_window = (window_size - n_tokens_out_reduced_window) / window_size
            n_tokens_out_reduced_window_list.append(n_tokens_out_reduced_window)
            compression_ratios_window_list.append(compression_ratio_window)

        # Write results
        df = pl.DataFrame(
            {
                "model_name": [model_name],
                "temperature": [temperature],
                "window_size": [window_size],
                "n_tokens_in": [n_tokens_in],
                "n_tokens_out": [n_tokens_out],
                "n_tokens_reduced": [n_tokens_out_reduced],
                "compression_ratio": [compression_ratio],
                "n_tokens_out_reduced_window": [str(n_tokens_out_reduced_window_list)],
                "compression_ratio_window": [str(compression_ratios_window_list)],
                "prompt": [tokenizer.batch_decode(input_ids, skip_special_tokens=True)[0]],
                "completion": [tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]],
                "output_ids": [str(output_ids.squeeze().tolist())],
                "output_ids_compressed": [str(output_ids_compressed.squeeze().tolist())],
            }
        )
        if os.path.isfile(savefile):
            with open(savefile, 'ab') as f:
                df.write_csv(f, include_header=False)
        else:
            df.write_csv(savefile)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", 
        choices=list(PYTHIA_MODELS.keys()) +["all"], 
        default="pythia-70m",
        nargs="+",
    )
    parser.add_argument(
        "--except-models", 
        choices=list(PYTHIA_MODELS.keys()), 
        default="pythia-12b",
        nargs="+",
    )
    parser.add_argument("--n-texts", type=int, default=128)
    parser.add_argument("--savefile", type=str, default="results.csv")
    parser.add_argument("--max-gen-tokens", type=int, default=1024)
    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0, nargs="+")
    parser.add_argument("--deduped", type=int, default=0)
    parser.add_argument("--compile", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.models in ("all", ["all"]):
        model_names = list(PYTHIA_MODELS.keys())
    elif isinstance(args.models, str):
        model_names = [args.models]
    else:
        model_names = args.models

    except_model_names = [args.except_models] if isinstance(args.except_models, str) else args.except_models
    model_names = [m for m in model_names if m not in except_model_names]
    model_names = list(set(model_names))

    temperatures = args.temperature if isinstance(args.temperature, list) else [args.temperature]
    temperatures = list(set(temperatures))

    last_model_name = ""
    loop = tqdm(
        list(itertools.product(model_names, temperatures)), 
        total=len(model_names) * len(temperatures),
    )
    for i, (model_name, temperature) in enumerate(loop):
        if model_name != last_model_name:
            model, tokenizer = load_model_and_tokenizer(model_name, args.deduped)
            dataset = load_and_tokenize_wikitext(tokenizer, n_texts=args.n_texts)
            if compile:
                model = torch.compile(model)
            model = model.to(device)
            last_model_name = model_name


        measure_compression(
            model_name=model_name,
            model=model, 
            tokenizer=tokenizer, 
            dataset=dataset,
            savefile=args.savefile,
            max_gen_tokens=args.max_gen_tokens,
            window_size=args.window_size,
            temperature=temperature,
            loop=loop,
            device=device,
        )

        df = pl.scan_csv(args.savefile).filter(
            (pl.col("model_name") == model_name) 
            & (pl.col("temperature") == temperature)
        ).collect()
        num_compressed = len(df.filter(pl.col("compression_ratio") > 0.0))
        percent_compressed = num_compressed / len(df) * 100
        loop.write(
            f"({i+1}/{len(model_names) * len(temperatures)}) "
            f"model={model_name}, "
            f"temperature={temperature}; "
            f"compression: "
            f"mean={df['compression_ratio'].mean():.5f}, "
            f"std={df['compression_ratio'].std():.5f}, "
            f"min={df['compression_ratio'].min():.5f}, "
            f"max={df['compression_ratio'].max():.5f}, "
            f"num={num_compressed}, "
            f"percent={percent_compressed:.2f}%"
        )


if __name__ == "__main__":
    main()
