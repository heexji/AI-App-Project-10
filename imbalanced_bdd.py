import os
import json
import random
from collections import Counter

from PIL import Image
from torch.utils.data import Dataset


class IMBALANCEBDDWeather(Dataset):
  

    def __init__(
        self,
        base_dir,
        split="train",      # "train" or "val"
        imb_type="exp",     # "exp", "step", "none"
        imb_factor=0.01,
        transform=None,
        rand_number=0,
    ):
        self.transform = transform
        self.split = split
        random.seed(rand_number)

        # 1) 라벨 JSON 경로 (labels 쪽은 이미 정상 동작하던 구조 그대로)
        if split == "train":
            json_path = os.path.join(
                base_dir, "bdd100k_labels_release", "bdd100k",
                "labels", "bdd100k_labels_images_train.json",
            )
        elif split == "val":
            json_path = os.path.join(
                base_dir, "bdd100k_labels_release", "bdd100k",
                "labels", "bdd100k_labels_images_val.json",
            )

        else:
            raise ValueError(f"Unknown split: {split}")

        # 2) 이미지 루트: 100k/{train,val} 아래 모든 하위 폴더를 스캔
        if split == "train":
            img_root = os.path.join(
                base_dir, "bdd100k", "bdd100k", "images", "100k", "train"
            )
        else:  # val
            img_root = os.path.join(
                base_dir, "bdd100k", "bdd100k", "images", "100k", "val"
            )

        # 파일 이름(basename) -> 전체 경로 dict 생성
        # (trainA/trainB/testA/testB 등 하위 폴더 신경 쓸 필요 없음)
        name2path = {}
        for root, dirs, files in os.walk(img_root):
            for fname in files:
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    name2path[fname] = os.path.join(root, fname)

        # 3) JSON에서 (img_path, label) 만들기 (weather 사용)
        with open(json_path, "r") as f:
            meta = json.load(f)

        all_weathers = sorted({m["attributes"]["weather"] for m in meta})
        self.weather_to_idx = {w: i for i, w in enumerate(all_weathers)}
        self.num_classes = len(self.weather_to_idx)

        samples = []
        missing = 0
        for m in meta:
            w = m["attributes"]["weather"]
            if w not in self.weather_to_idx:
                continue
            label = self.weather_to_idx[w]
            name = m["name"]                      # 예: "6891f028-b57c3dd4.jpg"
            path = name2path.get(name)
            if path is None:
                missing += 1
                continue
            samples.append((path, label))

        if missing > 0:
            print(f"[WARN] {missing} images not found under {img_root}")

        labels = [y for _, y in samples]
        orig_cls_num_list = self._get_cls_num_list(labels)

        # 4) long-tail 서브샘플링
        if imb_type.lower() == "none":
            self.samples = samples
        else:
            self.samples = self._gen_imbalanced_data(
                samples, orig_cls_num_list, imb_type, imb_factor
            )

        final_labels = [y for _, y in self.samples]
        self.cls_num_list = self._get_cls_num_list(final_labels)

    # ---------- util ----------
    def _get_cls_num_list(self, labels):
        cnt = Counter(labels)
        return [cnt[i] for i in range(self.num_classes)]

    def get_cls_num_list(self):
        return self.cls_num_list

    def _gen_imbalanced_data(self, samples, cls_num_list, imb_type, imb_factor):
        max_num = max(cls_num_list)
        num_classes = len(cls_num_list)

        if imb_type == "exp":
            img_num_per_cls = [
                int(max_num * (imb_factor ** (i / (num_classes - 1.0))))
                for i in range(num_classes)
            ]
        elif imb_type == "step":
            img_num_per_cls = []
            for i in range(num_classes):
                if i < num_classes / 2:
                    img_num_per_cls.append(max_num)
                else:
                    img_num_per_cls.append(int(max_num * imb_factor))
        else:
            return samples

        samples_by_cls = [[] for _ in range(num_classes)]
        for path, label in samples:
            samples_by_cls[label].append((path, label))

        new_samples = []
        for cls_idx, cls_samples in enumerate(samples_by_cls):
            n = min(img_num_per_cls[cls_idx], len(cls_samples))
            if n > 0:
                new_samples.extend(random.sample(cls_samples, n))
        return new_samples

    # ---------- Dataset ----------
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label
