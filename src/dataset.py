import os
import glob
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from scipy import signal
from scipy.io import wavfile
import cv2
from PIL import Image
import numpy as np


class Synth90kDataset(Dataset):
    CHARS = '0123456789abcdefghijklmnopqrstuvwxyz'
    CHAR2LABEL = {char: i + 1 for i, char in enumerate(CHARS)}
    LABEL2CHAR = {label: char for char, label in CHAR2LABEL.items()}

    def __init__(self, root_dir=None, mode=None, paths=None, img_height=32, img_width=100):
        if root_dir and mode and not paths:
            paths, texts = self._load_from_raw_files(root_dir, mode)
        elif not root_dir and not mode and paths:
            texts = None

        self.paths = paths
        self.texts = texts
        self.img_height = img_height
        self.img_width = img_width

    def _load_from_raw_files(self, root_dir, mode):
        mapping = {}
        with open(os.path.join(root_dir, 'lexicon.txt'), 'r') as fr:
            for i, line in enumerate(fr.readlines()):
                mapping[i] = line.strip()

        paths_file = None
        if mode == 'train':
            paths_file = 'annotation_train.txt'
        elif mode == 'dev':
            paths_file = 'annotation_val.txt'
        elif mode == 'test':
            paths_file = 'annotation_test.txt'

        paths = []
        texts = []
        with open(os.path.join(root_dir, paths_file), 'r') as fr:
            for line in fr.readlines():
                path, index_str = line.strip().split(' ')
                path = os.path.join(root_dir, path)
                index = int(index_str)
                text = mapping[index]
                paths.append(path)
                texts.append(text)
        return paths, texts

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        try:
            image = Image.open(path).convert('L')  # grey-scale
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0

        image = torch.FloatTensor(image)
        if self.texts:
            text = self.texts[index]
            target = [self.CHAR2LABEL[c] for c in text]
            target_length = [len(target)]

            target = torch.LongTensor(target)
            target_length = torch.LongTensor(target_length)
            return image, target, target_length
        else:
            return image


def synth90k_collate_fn(batch):
    images, targets, target_lengths = zip(*batch)
    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0)
    target_lengths = torch.cat(target_lengths, 0)
    return images, targets, target_lengths


def load_synthetic_ocr_news_entries(labels_path):
    with open(labels_path, 'r', encoding='utf-8') as file_handle:
        data = json.load(file_handle)
    if not isinstance(data, list):
        raise ValueError("labels.json must be a list of {image_path, label} objects.")
    return data


def build_charset(texts):
    charset = sorted(set(''.join(texts)))
    return charset


def save_charset(path, charset):
    with open(path, 'w', encoding='utf-8') as file_handle:
        json.dump(charset, file_handle, ensure_ascii=False, indent=2)


def load_charset(path):
    with open(path, 'r', encoding='utf-8') as file_handle:
        charset = json.load(file_handle)
    if not isinstance(charset, list):
        raise ValueError("Charset file must be a JSON list of characters.")
    return charset


def resolve_synthetic_ocr_news_path(data_dir, image_path):
    data_dir_path = Path(data_dir)
    image_path_obj = Path(image_path)
    if image_path_obj.is_absolute():
        return str(image_path_obj)

    candidate = data_dir_path / image_path_obj
    if candidate.exists():
        return str(candidate)

    if image_path_obj.parts and image_path_obj.parts[0] == data_dir_path.name:
        candidate = data_dir_path / Path(*image_path_obj.parts[1:])
        if candidate.exists():
            return str(candidate)

    candidate = data_dir_path / "images" / image_path_obj.name
    return str(candidate)


class SyntheticOCRNewsDataset(Dataset):
    def __init__(
        self,
        data_dir=None,
        labels_file="labels.json",
        paths=None,
        texts=None,
        charset=None,
        img_height=32,
        img_width=100,
        max_label_length=None,
    ):
        if paths is not None:
            if texts is not None and len(paths) != len(texts):
                raise ValueError("paths and texts must have the same length.")
            self.paths = [str(path) for path in paths]
            self.texts = texts
        else:
            if data_dir is None:
                raise ValueError("data_dir is required when paths are not provided.")
            labels_path = os.path.join(data_dir, labels_file)
            entries = load_synthetic_ocr_news_entries(labels_path)
            self.paths = []
            self.texts = []
            for entry in entries:
                path = resolve_synthetic_ocr_news_path(data_dir, entry["image_path"])
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Missing image file: {path}")
                label = entry["label"]
                if max_label_length and len(label) > max_label_length:
                    continue
                self.paths.append(path)
                self.texts.append(label)

        self.img_height = img_height
        self.img_width = img_width
        if charset is None and self.texts is not None:
            charset = build_charset(self.texts)
        self.charset = charset
        if charset:
            self.char2label = {char: i + 1 for i, char in enumerate(charset)}
            self.label2char = {label: char for char, label in self.char2label.items()}
        else:
            self.char2label = None
            self.label2char = None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        try:
            image = Image.open(path).convert('L')
        except IOError:
            print('Corrupted image for %d' % index)
            return self[index + 1]

        image = image.resize((self.img_width, self.img_height), resample=Image.BILINEAR)
        image = np.array(image)
        image = image.reshape((1, self.img_height, self.img_width))
        image = (image / 127.5) - 1.0
        image = torch.FloatTensor(image)

        if self.texts is None:
            return image

        if not self.char2label:
            raise ValueError("charset must be provided to encode labels.")

        text = self.texts[index]
        target = [self.char2label[c] for c in text]
        target_length = [len(target)]

        target = torch.LongTensor(target)
        target_length = torch.LongTensor(target_length)
        return image, target, target_length
