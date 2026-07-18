import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


class JsonlImageDataset(Dataset):
    def __init__(self, dataset_dir, tokenizer, resolution):
        self.dataset_dir = Path(dataset_dir)
        self.tokenizer = tokenizer
        self.records = [
            json.loads(line)
            for line in (self.dataset_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.image_transform = transforms.Compose(
            [
                transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(self.dataset_dir / record["file_name"]).convert("RGB")
        text = record.get("text") or record.get("caption") or record.get("prompt")
        tokenized = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        return {
            "pixel_values": self.image_transform(image),
            "input_ids": tokenized.input_ids[0],
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="outputs/lora_ink")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--max-train-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="fp16" if torch.cuda.is_available() else "no",
    )
    torch.manual_seed(args.seed)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = CLIPTokenizer.from_pretrained(args.model, subfolder="tokenizer")
    load_kwargs = {"variant": "fp16", "use_safetensors": True} if torch.cuda.is_available() else {}
    text_encoder = CLIPTextModel.from_pretrained(args.model, subfolder="text_encoder", **load_kwargs)
    vae = AutoencoderKL.from_pretrained(args.model, subfolder="vae", **load_kwargs)
    unet = UNet2DConditionModel.from_pretrained(args.model, subfolder="unet", **load_kwargs)
    noise_scheduler = DDPMScheduler.from_pretrained(args.model, subfolder="scheduler")

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_config)
    unet.enable_gradient_checkpointing()

    dataset = JsonlImageDataset(args.dataset, tokenizer, args.resolution)
    dataloader = DataLoader(dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=2)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unet.parameters()), lr=args.learning_rate)

    unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    vae.to(accelerator.device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    text_encoder.to(accelerator.device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    vae.eval()
    text_encoder.eval()

    progress = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    global_step = 0
    while global_step < args.max_train_steps:
        for batch in dataloader:
            with accelerator.accumulate(unet):
                pixel_values = batch["pixel_values"].to(accelerator.device)
                input_ids = batch["input_ids"].to(accelerator.device)
                weight_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

                with torch.no_grad():
                    latents = vae.encode(pixel_values.to(dtype=weight_dtype)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                    encoder_hidden_states = text_encoder(input_ids)[0]

                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=latents.device,
                ).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                target = noise
                if noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress.update(1)
                global_step += 1
                progress.set_postfix(loss=f"{loss.detach().item():.4f}")
                if global_step >= args.max_train_steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_unet = accelerator.unwrap_model(unet)
        StableDiffusionPipeline.save_lora_weights(
            save_directory=output,
            unet_lora_layers=get_peft_model_state_dict(unwrapped_unet),
            safe_serialization=True,
        )
        (output / "train_args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
        print(f"saved LoRA to {output}")


if __name__ == "__main__":
    main()
