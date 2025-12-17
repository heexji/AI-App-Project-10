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

# ===================== 설정 =====================
DATA_ROOT = "./data"
BATCH_SIZE = 128
NUM_CLASSES = 10
NUM_WORKERS = 4

CHECKPOINT_PATH = "checkpoints_ce/best_ce_cifar10_resnet32.pth"
OUTPUT_CSV = "ce_eval_full.csv"
OUTPUT_ANOMALY_CSV = "ce_anomaly_eval.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        return self.linear(out.view(out.size(0), -1))


def resnet32(num_classes=10):
    return ResNet_CIFAR(BasicBlock, [5, 5, 5], num_classes)


# ===================== 데이터 =====================
def get_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )


def get_cifar10():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    return torchvision.datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=tf)


def get_cifar100():
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
    ])
    return torchvision.datasets.CIFAR100(DATA_ROOT, train=False, download=True, transform=tf)


# ===================== 유틸 =====================
def classification_metrics(y_true, y_pred):
    acc = (y_true == y_pred).mean()
    precision = []
    recall = []
    f1 = []
    for c in range(NUM_CLASSES):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        precision.append(p)
        recall.append(r)
        f1.append(0 if p+r==0 else 2*p*r/(p+r))
    return acc, np.mean(precision), np.mean(recall), np.mean(f1)


def anomaly_metrics(in_scores, out_scores):
    scores = np.concatenate([in_scores, out_scores])
    labels = np.concatenate([np.zeros(len(in_scores)), np.ones(len(out_scores))])
    qs = np.linspace(0, 1, 2001)
    best = {"gap":1e9}
    for q in qs:
        thr = np.quantile(scores, q)
        pred = (scores >= thr).astype(int)
        fp = np.sum((pred==1)&(labels==0))
        fn = np.sum((pred==0)&(labels==1))
        far = fp / max(np.sum(labels==0),1)
        frr = fn / max(np.sum(labels==1),1)
        if abs(far-frr) < best["gap"]:
            f1 = 2*np.sum((pred==1)&(labels==1)) / max(2*np.sum((pred==1)&(labels==1))+fp+fn,1)
            best = dict(thr=thr, FAR=far, FRR=frr, EER=(far+frr)/2, F1=f1, gap=abs(far-frr))
    return best


# ===================== main =====================
def main():
    model = resnet32(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE)["model_state"])
    model.eval()

    # ---- classification ----
    cifar10 = get_cifar10()
    loader = get_loader(cifar10)
    y_true, y_pred = [], []
    in_scores = []

    with torch.no_grad():
        for x,y in loader:
            x = x.to(DEVICE)
            logits = model(x)
            prob = F.softmax(logits,1)
            y_pred.extend(prob.argmax(1).cpu().numpy())
            y_true.extend(y.numpy())
            in_scores.extend((1-prob.max(1).values).cpu().numpy())

    acc, prec, rec, f1 = classification_metrics(np.array(y_true), np.array(y_pred))

    # ---- anomaly ----
    cifar100 = get_cifar100()
    loader_out = get_loader(cifar100)
    out_scores = []

    with torch.no_grad():
        for x,_ in loader_out:
            x = x.to(DEVICE)
            prob = F.softmax(model(x),1)
            out_scores.extend((1-prob.max(1).values).cpu().numpy())

    anomaly = anomaly_metrics(np.array(in_scores), np.array(out_scores))

    # ---- save ----
    with open(OUTPUT_CSV,"w",newline="") as f:
        w=csv.writer(f)
        w.writerow(["Accuracy","Precision","Recall","F1"])
        w.writerow([acc,prec,rec,f1])

    with open(OUTPUT_ANOMALY_CSV,"w",newline="") as f:
        w=csv.writer(f)
        for k,v in anomaly.items():
            w.writerow([k,v])

    print("CE evaluation done.")


if __name__=="__main__":
    main()
