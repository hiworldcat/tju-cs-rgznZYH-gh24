import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from generate import build_pipeline


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--negative-prompt", default="low quality, blurry, text, watermark")
    parser.add_argument("--out", default="outputs/compare")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", default="20,30,40")
    parser.add_argument("--guidance-scales", default="6.5,8.0,10.0")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.prompt_file:
        args.prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()
    if not args.prompt:
        raise ValueError("Provide --prompt or --prompt-file")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pipe = build_pipeline(args.model)
    steps_list = [int(x) for x in args.steps.split(",")]
    guidance_list = [float(x) for x in args.guidance_scales.split(",")]
    records = []
    thumbs = []

    for steps in steps_list:
        for guidance in guidance_list:
            generator = torch.Generator(device=pipe.device).manual_seed(args.seed)
            image = pipe(
                prompt=args.prompt,
                negative_prompt=args.negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator,
                height=512,
                width=512,
            ).images[0]
            name = f"steps{steps}_guidance{guidance}.png"
            image.save(out_dir / name)
            records.append({"file_name": name, "steps": steps, "guidance_scale": guidance})

            thumb = image.resize((256, 256))
            canvas = Image.new("RGB", (256, 286), "white")
            canvas.paste(thumb, (0, 0))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 262), f"steps={steps}, cfg={guidance}", fill=(0, 0, 0))
            thumbs.append(canvas)

    grid = Image.new("RGB", (256 * len(guidance_list), 286 * len(steps_list)), "white")
    for i, thumb in enumerate(thumbs):
        x = (i % len(guidance_list)) * 256
        y = (i // len(guidance_list)) * 286
        grid.paste(thumb, (x, y))
    grid.save(out_dir / "grid.png")
    (out_dir / "records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(out_dir / "grid.png")


if __name__ == "__main__":
    main()
