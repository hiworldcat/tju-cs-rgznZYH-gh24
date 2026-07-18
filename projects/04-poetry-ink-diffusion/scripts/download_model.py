import argparse
import os

from huggingface_hub import snapshot_download


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    parser.add_argument("--local-dir", default="models/stable-diffusion-v1-5")
    parser.add_argument("--revision", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=args.local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
    print(path)


if __name__ == "__main__":
    main()
