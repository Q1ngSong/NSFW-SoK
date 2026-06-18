import os
import pandas as pd

from PIL import Image
import argparse
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
import numpy as np
import re
import torch


def sorted_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split("([0-9]+)", key)]
    return sorted(l, key=alphanum_key)


def normalize_promptid(value):
    text = str(value).strip()
    if text.endswith(".0"):
        return text[:-2]
    return text


def clip_score_cumstom(folder_path, prompts_path, model_version="large", batch_size=32):
    if model_version == "base":
        model_path = "/root/hf_models/openai/clip-vit-base-patch32"
    elif model_version == "large":
        model_path = "/root/hf_models/openai/clip-vit-large-patch14"
    else:
        raise ValueError(f"Unknown CLIP model version: {model_version}")

    model = CLIPModel.from_pretrained(model_path)
    processor = CLIPProcessor.from_pretrained(model_path)

    df = pd.read_csv(prompts_path)
    if "promptid" not in df.columns:
        raise KeyError("promptid")
    if "prompt" not in df.columns:
        raise KeyError("prompt")

    df["promptid"] = df["promptid"].apply(normalize_promptid)
    prompt_by_id = dict(zip(df["promptid"], df["prompt"]))
    images = sorted_nicely(os.listdir(folder_path))

    ratios = {}
    df["clip"] = np.nan

    for image in tqdm(images):
        if not image.endswith(".png"):
            continue
        promptid = normalize_promptid(image.split("_")[0].replace(".png", ""))
        if promptid not in prompt_by_id:
            continue
        caption = prompt_by_id[promptid]
        im = Image.open(os.path.join(folder_path, image)).convert("RGB")
        inputs = processor(text=[caption], images=im, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        clip_score = outputs.logits_per_image[0][0].detach().cpu().item()
        ratios[promptid] = ratios.get(promptid, []) + [clip_score]

    for promptid, scores in ratios.items():
        df.loc[df.promptid == promptid, "clip"] = float(np.mean(scores))

    df = df.dropna(axis=0)
    avg_score = float(df["clip"].mean()) if len(df) else float("nan")
    print(f"Mean CLIP score: {avg_score}")
    print("-------------------------------------------------")
    print("\n")
    return df, avg_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, default="adv_images")
    parser.add_argument("--model_version", type=str, choices=["base", "large"], default="base")
    parser.add_argument("--prompts_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=96)
    args = parser.parse_args()
    folder_path = args.folder_path
    prompts_path = args.prompts_path
    batch_size = args.batch_size
    if prompts_path is None:
        prompts_path = os.path.join(folder_path, "prompts.csv")
    if not os.path.exists(prompts_path):
        raise Exception("prompts_path not exists")

    if os.path.exists(os.path.join(folder_path, "imgs")):
        img_folder = os.path.join(folder_path, "imgs")
    else:
        img_folder = folder_path

    clip_df, avg_clip_score = clip_score_cumstom(img_folder, prompts_path, args.model_version, batch_size)
    clip_df = pd.concat(
        [clip_df, pd.DataFrame([{"promptid": "average", "clip": avg_clip_score}])],
        ignore_index=True,
    )
    clip_df.to_csv(os.path.join(folder_path, "clip_scores.csv"), index=False)
