import argparse
import time
from pathlib import Path

import mindspore as ms

from src.data import ImageFolderDataset, find_garbage_root
from src.models import LeNet5, LeNet5BN, count_parameters, freeze_feature_extractor
from src.train_utils import make_dataset, train_loop, write_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LeNet-5 on the 26-class garbage dataset.")
    parser.add_argument("--data-root", default="data/data_en")
    parser.add_argument("--pretrained", default="checkpoints/lenet5_mnist.ckpt")
    parser.add_argument("--strategy", choices=["frozen", "full", "scratch"], default="frozen")
    parser.add_argument("--variant", choices=["lenet5", "lenet5_bn", "lenet5_color"], default="lenet5")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=32)
    return parser.parse_args()


def load_backbone_from_mnist(net, ckpt_path):
    ckpt = Path(ckpt_path)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"MNIST checkpoint not found: {ckpt}. Run train_mnist.py first, "
            "or use --strategy scratch."
        )
    params = ms.load_checkpoint(str(ckpt))
    reusable = {
        name: value
        for name, value in params.items()
        if not name.startswith("classifier.4.")
    }
    not_loaded = ms.load_param_into_net(net, reusable, strict_load=False)
    print(f"loaded pretrained params except final classifier: {len(reusable)} tensors")
    print(f"not loaded: {not_loaded}")


def main():
    args = parse_args()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")

    root = find_garbage_root(args.data_root)
    use_grayscale = args.variant != "lenet5_color"
    train_data = ImageFolderDataset(
        root / "train", image_size=args.image_size, grayscale=use_grayscale, augment=True
    )
    test_data = ImageFolderDataset(
        root / "test", image_size=args.image_size, grayscale=use_grayscale, augment=False
    )
    if train_data.classes != test_data.classes:
        raise RuntimeError("Train/test class lists differ.")
    num_classes = len(train_data.classes)
    print(f"dataset root: {root}")
    print(f"classes: {num_classes}")
    print(", ".join(train_data.classes))

    if args.variant == "lenet5_bn":
        if args.strategy != "scratch":
            raise ValueError("lenet5_bn is intended for scratch model-improvement comparison.")
        net = LeNet5BN(num_classes=num_classes)
    elif args.variant == "lenet5_color":
        if args.strategy != "scratch":
            raise ValueError("lenet5_color changes the first convolution and is compared from scratch.")
        net = LeNet5(num_classes=num_classes, in_channels=3)
    else:
        net = LeNet5(num_classes=num_classes)
    if args.strategy in {"frozen", "full"}:
        load_backbone_from_mnist(net, args.pretrained)
    if args.strategy == "frozen":
        freeze_feature_extractor(net)

    total, trainable = count_parameters(net)
    print(
        f"variant={args.variant} strategy={args.strategy} "
        f"total_params={total} trainable_params={trainable}"
    )
    train_ds = make_dataset(train_data, args.batch_size, shuffle=True)
    test_ds = make_dataset(test_data, args.batch_size, shuffle=False)

    name = f"{args.variant}_garbage_{args.strategy}"
    ckpt_path = Path("checkpoints") / f"{name}.ckpt"
    start = time.perf_counter()
    history = train_loop(net, train_ds, test_ds, args.epochs, args.lr, str(ckpt_path))
    for row in history:
        row["strategy"] = args.strategy
        row["variant"] = args.variant
        row["total_params"] = total
        row["trainable_params"] = trainable
    write_csv(Path("results") / f"{name}.csv", history)
    print(f"best checkpoint: {ckpt_path}")
    print(f"total runtime: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
