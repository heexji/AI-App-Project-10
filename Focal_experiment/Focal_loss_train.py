import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import math
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# PIL import를 dataset보다 상단으로 이동 (중요!)
from PIL import Image

# ===================== 설정 =====================
DATA_ROOT = "./data"
IMB_TYPE = "long-tailed"    # "long-tailed" or "step"
IMB_FACTOR = 0.01           # long-tailed 비율 100
NUM_CLASSES = 10

BATCH_SIZE = 128
EPOCHS = 200
WARMUP_EPOCHS = 5
MILESTONES = [160, 180]
BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 2e-4
NUM_WORKERS = 4

FOCAL_GAMMA = 2.0
USE_ALPHA = False  # 클래스별 가중치 사용 여부

SEED = 42
# =================================================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ===================== ResNet-32 (CIFAR용) =====================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


class ResNet_CIFAR(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet_CIFAR, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(
            3, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        feat = out.view(out.size(0), -1)
        out = self.linear(feat)
        if return_feature:
            return out, feat
        return out


def resnet32(num_classes=10):
    return ResNet_CIFAR(BasicBlock, [5, 5, 5], num_classes=num_classes)


# ===================== Imbalanced CIFAR-10 =====================
class ImbalancedCIFAR10(Dataset):
    def __init__(
        self,
        root,
        train=True,
        imb_type="long-tailed",
        imb_factor=0.01,
        download=True,
        transform=None,
    ):
        self.train = train
        self.imb_type = imb_type
        self.imb_factor = imb_factor
        self.transform = transform

        cifar10 = torchvision.datasets.CIFAR10(
            root=root, train=train, download=download
        )
        self.data = cifar10.data
        self.targets = np.array(cifar10.targets)

        if self.train:
            self.gen_imbalanced_data()

    def get_img_num_per_cls(self, cls_num: int) -> List[int]:
        img_max = len(self.targets) / cls_num

        if self.imb_type == "long-tailed":
            img_num_per_cls = []
            for cls_idx in range(cls_num):
                num = img_max * (self.imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif self.imb_type == "step":
            img_num_per_cls = []
            for cls_idx in range(cls_num):
                if cls_idx < cls_num / 2:
                    img_num_per_cls.append(int(img_max))
                else:
                    img_num_per_cls.append(int(img_max * self.imb_factor))
        else:
            img_num_per_cls = [int(img_max)] * cls_num
        return img_num_per_cls

    def gen_imbalanced_data(self):
        cls_num = NUM_CLASSES
        img_num_per_cls = self.get_img_num_per_cls(cls_num)
        new_data = []
        new_targets = []

        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)

        for cls_idx, cls in enumerate(classes):
            cls_mask = np.where(targets_np == cls)[0]
            np.random.shuffle(cls_mask)
            selec_idx = cls_mask[: img_num_per_cls[cls_idx]]
            new_data.append(self.data[selec_idx, ...])
            new_targets.extend([cls] * len(selec_idx))

        self.data = np.vstack(new_data)
        self.targets = np.array(new_targets, dtype=np.int64)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target


# ===================== Focal Loss =====================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if isinstance(alpha, (list, np.ndarray)):
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none",
            weight=self.alpha.to(logits.device) if isinstance(self.alpha, torch.Tensor) else None
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ===================== 데이터 로더 =====================
def get_dataloaders():
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = ImbalancedCIFAR10(
        root=DATA_ROOT,
        train=True,
        imb_type=IMB_TYPE,
        imb_factor=IMB_FACTOR,
        download=True,
        transform=train_transform,
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root=DATA_ROOT,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, test_loader, train_dataset


# ===================== 학습 루프 =====================
def adjust_learning_rate(optimizer, epoch, base_lr):
    if epoch < WARMUP_EPOCHS:
        lr = base_lr * float(epoch + 1) / WARMUP_EPOCHS
    else:
        lr = base_lr
        for m in MILESTONES:
            if epoch >= m:
                lr *= 0.01
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


def train_one_epoch(model, criterion, optimizer, train_loader, device, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    lr = adjust_learning_rate(optimizer, epoch, BASE_LR)

    for images, targets in train_loader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc, lr


def evaluate(model, criterion, test_loader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            loss = criterion(logits, targets)

            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    avg_loss = total_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


def compute_class_counts(dataset):
    targets = np.array(dataset.targets)
    counts = [np.sum(targets == i) for i in range(NUM_CLASSES)]
    return counts


def main():
    set_seed(SEED)

    # 이 device 설정은 CUDA_VISIBLE_DEVICES=1 덕에 GPU1만 보임
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader, train_dataset = get_dataloaders()
    cls_counts = compute_class_counts(train_dataset)
    print("Class counts (train, imbalanced):", cls_counts)

    alpha = None
    if USE_ALPHA:
        cls_counts_np = np.array(cls_counts, dtype=np.float32)
        inv = 1.0 / cls_counts_np
        alpha = inv / inv.sum() * NUM_CLASSES
        print("Using alpha (per-class weights):", alpha)

    model = resnet32(num_classes=NUM_CLASSES).to(device)

    criterion = FocalLoss(
        gamma=FOCAL_GAMMA,
        alpha=alpha,
        reduction="mean",
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=BASE_LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=False,
    )

    best_acc = 0.0
    os.makedirs("model", exist_ok=True)

    for epoch in range(EPOCHS):
        train_loss, train_acc, lr = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch
        )
        test_loss, test_acc = evaluate(model, criterion, test_loader, device)

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"lr={lr:.5f} "
            f"Train Loss={train_loss:.4f} Acc={train_acc:.2f}% "
            f"Test Loss={test_loss:.4f} Acc={test_acc:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_acc": best_acc,
                },
                os.path.join("model", "best_focal_cifar10_resnet32.pth"),
            )

    print(f"Best Test Acc: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
