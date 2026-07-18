import csv
import time
from pathlib import Path

import mindspore as ms
import mindspore.nn as nn
from mindspore import ops
from tqdm import tqdm


def make_dataset(dataset, batch_size, shuffle=True):
    ds = ms.dataset.GeneratorDataset(dataset, ["image", "label"], shuffle=shuffle)
    return ds.batch(batch_size, drop_remainder=False)


def accuracy(logits, labels):
    preds = ops.Argmax(axis=1)(logits)
    return float((preds == labels).astype(ms.float32).mean().asnumpy())


def evaluate(net, dataset):
    net.set_train(False)
    total = 0
    correct = 0.0
    for image, label in dataset.create_tuple_iterator():
        logits = net(image)
        batch = int(label.shape[0])
        correct += accuracy(logits, label) * batch
        total += batch
    return correct / max(total, 1)


def train_loop(net, train_ds, test_ds, epochs, lr, ckpt_path=None):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)

    def forward_fn(image, label):
        logits = net(image)
        loss = loss_fn(logits, label)
        return loss

    grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters)

    def train_step(image, label):
        loss, grads = grad_fn(image, label)
        optimizer(grads)
        return loss

    history = []
    best_acc = -1.0
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        net.set_train(True)
        losses = []
        iterator = train_ds.create_tuple_iterator()
        for image, label in tqdm(iterator, desc=f"epoch {epoch}/{epochs}", leave=False):
            loss = train_step(image, label)
            losses.append(float(loss.asnumpy()))
        test_acc = evaluate(net, test_ds)
        avg_loss = sum(losses) / max(len(losses), 1)
        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "test_acc": test_acc,
            "elapsed_sec": time.perf_counter() - start,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} loss={avg_loss:.4f} "
            f"test_acc={test_acc:.4f} elapsed={row['elapsed_sec']:.1f}s"
        )
        if ckpt_path and test_acc > best_acc:
            best_acc = test_acc
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            ms.save_checkpoint(net, ckpt_path)
    return history


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
