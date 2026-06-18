import torch
from diffusers import StableDiffusionPipeline


def main():
    # 1. 模型路径：沿用你之前的本地 SD v1.5
    model_id = "/root/hf_models/stable-diffusion-v1-5/stable-diffusion-v1-5"

    # 2. 设备选择
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # 3. 加载 Stable Diffusion v1.5 pipeline
    #    这里只是为了拿 tokenizer 和 text_encoder，不做图像生成
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        dtype=torch.float16 if device == "cuda" else torch.float32,  # 新版 diffusers 推荐用 dtype
    )
    pipe = pipe.to(device)

    # 4. 定义 gore 相关的 prompt（拼写已修正）
    prompt = "an image showing gore, blood, bleeding, injury"

    # 5. 使用 pipeline 自带的 tokenizer 编码文本
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder

    text_inputs = tokenizer(
        [prompt],                              # batch size = 1
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )

    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device)

    # 6. 调用 text_encoder 得到文本 embedding
    #    与 Stable Diffusion 生成图像时使用的 text_embeddings 完全一致
    with torch.no_grad():
        encoder_outputs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # last_hidden_state: [batch_size, seq_len(=77), hidden_dim(=768)]
        text_embeddings = encoder_outputs.last_hidden_state

    # 7. 保存为 .pth 文件（在 CPU 上保存，方便后续加载）
    save_path = "./gore_embedding_sd15.pth"
    torch.save(text_embeddings.cpu(), save_path)

    print(f"[DONE] Saved embedding to: {save_path}")
    print(f"[INFO] Shape: {text_embeddings.shape}, dtype: {text_embeddings.dtype}")


if __name__ == "__main__":
    main()
