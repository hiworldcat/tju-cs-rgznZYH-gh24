import argparse
import json
import shutil
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFilter


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", default=None)
    parser.add_argument("--captions", default=None, help="CSV or parquet with file_name,caption")
    parser.add_argument("--output", required=True)
    parser.add_argument("--file-column", default="file_name")
    parser.add_argument("--caption-column", default="caption")
    parser.add_argument("--demo", action="store_true")
    return parser.parse_args()


def make_demo_dataset(output):
    output.mkdir(parents=True, exist_ok=True)
    captions = [
        ("ink_00.jpg", "Chinese ink wash painting of a lone boat on a snowy river, minimal monochrome"),
        ("ink_01.jpg", "Chinese ink wash mountain landscape, pine trees, mist and distant waterfall"),
        ("ink_02.jpg", "Chinese ink wash painting of spring rain over a quiet village"),
        ("ink_03.jpg", "Chinese ink wash painting, moonlit bamboo forest, soft paper texture"),
        ("ink_04.jpg", "realistic photo of a quiet lake at sunrise, soft fog, natural colors"),
        ("ink_05.jpg", "realistic photo of an old stone street after rain, detailed texture"),
    ]
    for index, (name, caption) in enumerate(captions):
        image = Image.new("RGB", (512, 512), (238, 236, 228))
        draw = ImageDraw.Draw(image)
        for i in range(18):
            shade = 45 + i * 8
            x0 = -40 + i * 32
            y0 = 360 - (i % 5) * 20
            draw.ellipse((x0, y0, x0 + 250, y0 + 120), fill=(shade, shade, shade))
        draw.line((30, 390, 480, 350 - index * 8), fill=(30, 30, 30), width=4)
        draw.text((24, 28), f"demo {index}", fill=(70, 70, 70))
        image = image.filter(ImageFilter.GaussianBlur(radius=0.6))
        image.save(output / name, quality=95)
    write_metadata(output, captions)


def read_caption_table(path):
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def write_metadata(output, rows):
    with (output / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for file_name, caption in rows:
            f.write(json.dumps({"file_name": file_name, "text": caption}, ensure_ascii=False) + "\n")


def convert_dataset(image_dir, captions, output, file_column, caption_column):
    image_dir = Path(image_dir)
    output.mkdir(parents=True, exist_ok=True)
    df = read_caption_table(captions)
    rows = []
    for _, row in df.iterrows():
        file_name = str(row[file_column])
        caption = str(row[caption_column])
        src = image_dir / file_name
        if not src.exists():
            raise FileNotFoundError(src)
        Image.open(src).convert("RGB").verify()
        dst = output / Path(file_name).name
        shutil.copy2(src, dst)
        rows.append((dst.name, caption))
    write_metadata(output, rows)
    print(f"wrote {len(rows)} samples to {output}")


def main():
    args = parse_args()
    output = Path(args.output)
    if args.demo:
        make_demo_dataset(output)
        print(f"wrote demo dataset to {output}")
        return
    if not args.image_dir or not args.captions:
        raise ValueError("Use --demo or provide --image-dir and --captions")
    convert_dataset(args.image_dir, args.captions, output, args.file_column, args.caption_column)


if __name__ == "__main__":
    main()
