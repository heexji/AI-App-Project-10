import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image

# ===================== 설정 =====================
DATA_ROOT = "./data"
BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_WORKERS = 4

CHECKPOINT_PATH = "model/best_focal_cifar10_resnet32.pth"
OUTPUT_CSV = "focal_eval_result.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===============================================


# ===================== ResNet-32 =====================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], 1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], 2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], 2)

        # ✅ 학습 코드와 동일하게 'linear'
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        return self.linear(out)


def resnet32(num_classes=10):
    return ResNet_CIFAR(BasicBlock, [5, 5, 5], num_classes)


# ===================== 데이터 =====================
def get_test_loader():
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=transform,
    )

    return DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


# ===================== 평가 =====================
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0

    class_correct = np.zeros(NUM_CLASSES)
    class_total = np.zeros(NUM_CLASSES)

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            correct += (preds == targets).sum().item()
            total += targets.size(0)

            for i in range(len(targets)):
                label = targets[i].item()
                class_correct[label] += (preds[i] == label).item()
                class_total[label] += 1

    overall_acc = 100.0 * correct / total
    per_class_acc = 100.0 * class_correct / class_total

    return overall_acc, per_class_acc


# ===================== main =====================
def main():
    print("Device:", DEVICE)

    model = resnet32(NUM_CLASSES).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    test_loader = get_test_loader()
    overall_acc, per_class_acc = evaluate(model, test_loader)

    print(f"Overall Test Accuracy: {overall_acc:.2f}%")
    print("Per-class Accuracy:", per_class_acc)

    # CSV 저장
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "accuracy"])
        for i, acc in enumerate(per_class_acc):
            writer.writerow([i, acc])
        writer.writerow(["overall", overall_acc])

    print(f"Saved evaluation results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
