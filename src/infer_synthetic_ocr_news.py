#!/usr/bin/env python3
import argparse
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import (
    SyntheticOCRNewsDataset,
    load_charset,
    load_synthetic_ocr_news_entries,
    resolve_synthetic_ocr_news_path,
)
from model import CRNN
from ctc_decoder import ctc_decode


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run inference on images for synthetic_ocr_news."
    )
    parser.add_argument("images", nargs="*", help="Image paths to predict.")
    parser.add_argument(
        "--data-dir",
        default="data/synthetic_ocr_news",
        help="Dataset root containing labels.json and images/.",
    )
    parser.add_argument(
        "--labels-file",
        default="labels.json",
        help="Labels file name inside data-dir (used when no images are provided).",
    )
    parser.add_argument(
        "--charset-path",
        default=None,
        help="Path to charset JSON (default: data-dir/charset.json).",
    )
    parser.add_argument(
        "--model",
        default="checkpoints/crnn_synthetic_ocr_news.pt",
        help="Checkpoint to load.",
    )
    parser.add_argument("--img-height", type=int, default=48)
    parser.add_argument("--img-width", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--decode-method",
        default="beam_search",
        choices=["greedy", "beam_search", "prefix_beam_search"],
    )
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-output", type=int, default=50)
    parser.add_argument("--map-to-seq-hidden", type=int, default=64)
    parser.add_argument("--rnn-hidden", type=int, default=256)
    parser.add_argument("--leaky-relu", action="store_true")
    return parser.parse_args()


def predict(crnn, dataloader, label2char, decode_method, beam_size):
    crnn.eval()
    pbar = tqdm(total=len(dataloader), desc="Predict")

    all_preds = []
    with torch.no_grad():
        for data in dataloader:
            device = "cuda" if next(crnn.parameters()).is_cuda else "cpu"

            images = data.to(device)
            logits = crnn(images)
            log_probs = torch.nn.functional.log_softmax(logits, dim=2)

            preds = ctc_decode(
                log_probs,
                method=decode_method,
                beam_size=beam_size,
                label2char=label2char,
            )
            all_preds.extend(preds)
            pbar.update(1)
        pbar.close()

    return all_preds


def edit_distance(text_a, text_b):
    if text_a == text_b:
        return 0
    if not text_a:
        return len(text_b)
    if not text_b:
        return len(text_a)

    prev = list(range(len(text_b) + 1))
    for i, char_a in enumerate(text_a, start=1):
        curr = [i]
        for j, char_b in enumerate(text_b, start=1):
            cost = 0 if char_a == char_b else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def char_accuracy(pred_texts, labels):
    total = 0.0
    for pred, label in zip(pred_texts, labels):
        denom = max(len(label), len(pred), 1)
        dist = edit_distance(pred, label)
        total += max(0.0, 1.0 - (dist / denom))
    return total / len(labels)


def main():
    args = parse_args()
    if args.img_height % 16 != 0:
        raise ValueError("img-height must be divisible by 16.")
    if args.img_width % 4 != 0:
        raise ValueError("img-width must be divisible by 4.")

    charset_path = args.charset_path or os.path.join(args.data_dir, "charset.json")
    charset = load_charset(charset_path)
    label2char = {i + 1: char for i, char in enumerate(charset)}

    images = list(args.images)
    labels = None
    if not images:
        labels_path = os.path.join(args.data_dir, args.labels_file)
        entries = load_synthetic_ocr_news_entries(labels_path)
        if args.limit:
            entries = entries[: args.limit]
        images = [
            resolve_synthetic_ocr_news_path(args.data_dir, entry["image_path"])
            for entry in entries
        ]
        labels = [entry["label"] for entry in entries]

    dataset = SyntheticOCRNewsDataset(
        paths=images,
        img_height=args.img_height,
        img_width=args.img_width,
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    num_class = len(charset) + 1
    crnn = CRNN(
        1,
        args.img_height,
        args.img_width,
        num_class,
        map_to_seq_hidden=args.map_to_seq_hidden,
        rnn_hidden=args.rnn_hidden,
        leaky_relu=args.leaky_relu,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    crnn.load_state_dict(torch.load(args.model, map_location=device))
    crnn.to(device)

    preds = predict(crnn, dataloader, label2char, args.decode_method, args.beam_size)
    pred_texts = ["".join(pred) for pred in preds]

    display_limit = len(images)
    if args.max_output and args.max_output > 0:
        display_limit = min(args.max_output, len(images))

    print("\n===== result =====")
    for i in range(display_limit):
        if labels:
            print(f"{images[i]} > {pred_texts[i]} | {labels[i]}")
        else:
            print(f"{images[i]} > {pred_texts[i]}")

    if labels:
        score = char_accuracy(pred_texts, labels)
        print(f"\nchar_accuracy: {score:.4f}")


if __name__ == "__main__":
    raise SystemExit(main())
