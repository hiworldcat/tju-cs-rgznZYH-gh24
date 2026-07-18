import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from diffusers import DPMSolverMultistepScheduler, StableDiffusionPipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--lora", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--prompts", default=None, help="CSV with id,prompt,negative_prompt columns")
    parser.add_argument("--out", default="outputs/base")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    return parser.parse_args()


def load_prompt_rows(args):
    if args.prompts:
        df = pd.read_csv(args.prompts)
        rows = []
        for idx, row in df.iterrows():
            rows.append(
                {
                    "id": str(row.get("id", idx)),
                    "prompt": row["prompt"],
                    "negative_prompt": row.get("negative_prompt", args.negative_prompt) or "",
                }
            )
        return rows
    if not args.prompt:
        raise ValueError("Provide --prompt or --prompts")
    return [{"id": "sample", "prompt": args.prompt, "negative_prompt": args.negative_prompt or ""}]


def build_pipeline(model, lora=None):
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model,
        torch_dtype=dtype,
        variant="fp16" if dtype == torch.float16 else None,
        use_safetensors=True,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if lora:
        pipe.load_lora_weights(lora)
    if torch.cuda.is_available():
        pipe.to("cuda")
        pipe.enable_attention_slicing()
    return pipe


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(args.model, args.lora)
    rows = load_prompt_rows(args)
    generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
    metadata_path = out_dir / "metadata.jsonl"

    with metadata_path.open("w", encoding="utf-8") as f:
        for row in rows:
            image = pipe(
                prompt=row["prompt"],
                negative_prompt=row["negative_prompt"],
                num_inference_steps=args.steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                generator=generator,
            ).images[0]
            file_name = f"{row['id']}_s{args.steps}_g{args.guidance_scale}_seed{args.seed}.png"
            image.save(out_dir / file_name)
            f.write(
                json.dumps(
                    {
                        "file_name": file_name,
                        "prompt": row["prompt"],
                        "negative_prompt": row["negative_prompt"],
                        "steps": args.steps,
                        "guidance_scale": args.guidance_scale,
                        "seed": args.seed,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            print(out_dir / file_name)


if __name__ == "__main__":
    main()
