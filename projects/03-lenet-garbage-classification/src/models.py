import mindspore.nn as nn


class LeNet5(nn.Cell):
    """LeNet-5 for 32x32 inputs."""

    def __init__(self, num_classes=10, in_channels=1):
        super().__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, pad_mode="valid"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, pad_mode="valid"),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.SequentialCell(
            nn.Dense(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Dense(120, 84),
            nn.ReLU(),
            nn.Dense(84, num_classes),
        )

    def construct(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)


class LeNet5BN(nn.Cell):
    """LeNet-5 with BatchNorm for the model-improvement comparison."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, pad_mode="valid"),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, pad_mode="valid"),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.classifier = nn.SequentialCell(
            nn.Dense(16 * 5 * 5, 120),
            nn.BatchNorm1d(120),
            nn.ReLU(),
            nn.Dense(120, 84),
            nn.BatchNorm1d(84),
            nn.ReLU(),
            nn.Dense(84, num_classes),
        )

    def construct(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)


def freeze_feature_extractor(net):
    for param in net.features.trainable_params():
        param.requires_grad = False


def count_parameters(net):
    total = 0
    trainable = 0
    for param in net.get_parameters():
        n = 1
        for dim in param.shape:
            n *= dim
        total += n
        if param.requires_grad:
            trainable += n
    return total, trainable
