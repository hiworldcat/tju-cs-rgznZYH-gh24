import argparse
from pathlib import Path

import mindspore as ms

from src.data import MnistArrayDataset, download_mnist
from src.models import LeNet5, count_parameters
from src.train_utils import make_dataset, train_loop, write_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain LeNet-5 on MNIST.")
    parser.add_argument("--data-dir", default="data/mnist")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-limit", type=int, default=20000)
    parser.add_argument("--test-limit", type=int, default=5000)
    parser.add_argument("--ckpt", default="checkpoints/lenet5_mnist.ckpt")
    return parser.parse_args()


def main():
    args = parse_args()
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    download_mnist(args.data_dir)

    train_data = MnistArrayDataset(args.data_dir, "train", limit=args.train_limit)
    test_data = MnistArrayDataset(args.data_dir, "test", limit=args.test_limit)
    train_ds = make_dataset(train_data, args.batch_size, shuffle=True)
    test_ds = make_dataset(test_data, args.batch_size, shuffle=False)

    net = LeNet5(num_classes=10)
    total, trainable = count_parameters(net)
    print(f"LeNet-5 parameters: total={total}, trainable={trainable}")
    history = train_loop(net, train_ds, test_ds, args.epochs, args.lr, args.ckpt)
    write_csv(Path("results") / "mnist_pretrain.csv", history)
    print(f"saved checkpoint: {args.ckpt}")


if __name__ == "__main__":
    main()
