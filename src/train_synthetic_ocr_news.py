#!/usr/bin/env python3
import argparse
import os

import torch
import torch.optim as optim
from torch.nn import CTCLoss
from torch.utils.data import DataLoader, random_split

from dataset import (
    SyntheticOCRNewsDataset,
    build_charset,
    load_charset,
    load_synthetic_ocr_news_entries,
    save_charset,
    synth90k_collate_fn,
)
from model import CRNN
from evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CRNN on the synthetic_ocr_news dataset."
    )
    parser.add_argument(
        "--data-dir",
        default="data/synthetic_ocr_news",
        help="Dataset root containing labels.json and images/.",
    )
    parser.add_argument(
        "--labels-file",
        default="labels.json",
        help="Labels file name inside data-dir.",
    )
    parser.add_argument(
        "--charset-path",
        default=None,
        help="Path to save/load charset JSON (default: data-dir/charset.json).",
    )
    parser.add_argument(
        "--overwrite-charset",
        action="store_true",
        help="Rebuild charset even if charset file already exists.",
    )
    parser.add_argument("--img-height", type=int, default=48)
    parser.add_argument("--img-width", type=int, default=320)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--show-interval", type=int, default=10)
    parser.add_argument("--valid-interval", type=int, default=200)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--cpu-workers", type=int, default=4)
    parser.add_argument("--valid-max-iter", type=int, default=100)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--decode-method",
        default="greedy",
        choices=["greedy", "beam_search", "prefix_beam_search"],
    )
    parser.add_argument("--beam-size", type=int, default=10)
    parser.add_argument(
        "--checkpoints-dir",
        default="checkpoints",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--reload-checkpoint",
        default=None,
        help="Optional path to a checkpoint to resume from.",
    )
    parser.add_argument("--map-to-seq-hidden", type=int, default=64)
    parser.add_argument("--rnn-hidden", type=int, default=256)
    parser.add_argument("--leaky-relu", action="store_true")
    return parser.parse_args()


def train_batch(crnn, data, optimizer, criterion, device):
    crnn.train()
    images, targets, target_lengths = [d.to(device) for d in data]

    logits = crnn(images)
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)

    batch_size = images.size(0)
    input_lengths = torch.LongTensor([logits.size(0)] * batch_size)
    target_lengths = torch.flatten(target_lengths)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(crnn.parameters(), 5)
    optimizer.step()
    return loss.item()


def main():
    args = parse_args()
    if args.img_height % 16 != 0:
        raise ValueError("img-height must be divisible by 16.")
    if args.img_width % 4 != 0:
        raise ValueError("img-width must be divisible by 4.")

    labels_path = os.path.join(args.data_dir, args.labels_file)
    entries = load_synthetic_ocr_news_entries(labels_path)
    texts = [entry["label"] for entry in entries]

    charset_path = args.charset_path or os.path.join(args.data_dir, "charset.json")
    if os.path.exists(charset_path) and not args.overwrite_charset:
        charset = load_charset(charset_path)
    else:
        charset = build_charset(texts)
        charset_dir = os.path.dirname(charset_path)
        if charset_dir:
            os.makedirs(charset_dir, exist_ok=True)
        save_charset(charset_path, charset)

    max_label_length = args.img_width // 4 - 1
    dataset = SyntheticOCRNewsDataset(
        data_dir=args.data_dir,
        labels_file=args.labels_file,
        charset=charset,
        img_height=args.img_height,
        img_width=args.img_width,
        max_label_length=max_label_length,
    )

    filtered = len(entries) - len(dataset)
    if filtered:
        print(f"filtered {filtered} samples exceeding max label length {max_label_length}")

    if args.val_ratio <= 0:
        train_dataset = dataset
        valid_dataset = None
    else:
        val_size = int(len(dataset) * args.val_ratio)
        val_size = max(1, val_size)
        train_size = len(dataset) - val_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, valid_dataset = random_split(
            dataset, [train_size, val_size], generator=generator
        )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.cpu_workers,
        collate_fn=synth90k_collate_fn,
    )
    valid_loader = None
    if valid_dataset is not None:
        valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=args.eval_batch_size,
            shuffle=True,
            num_workers=args.cpu_workers,
            collate_fn=synth90k_collate_fn,
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
    if args.reload_checkpoint:
        crnn.load_state_dict(torch.load(args.reload_checkpoint, map_location=device))
    crnn.to(device)

    optimizer = optim.RMSprop(crnn.parameters(), lr=args.lr)
    criterion = CTCLoss(reduction="sum", zero_infinity=True).to(device)

    os.makedirs(args.checkpoints_dir, exist_ok=True)
    if valid_loader is not None:
        assert args.save_interval % args.valid_interval == 0

    i = 1
    for epoch in range(1, args.epochs + 1):
        print(f"epoch: {epoch}")
        tot_train_loss = 0.0
        tot_train_count = 0
        for train_data in train_loader:
            loss = train_batch(crnn, train_data, optimizer, criterion, device)
            train_size = train_data[0].size(0)

            tot_train_loss += loss
            tot_train_count += train_size
            if i % args.show_interval == 0:
                print("train_batch_loss[", i, "]: ", loss / train_size)

            if valid_loader and i % args.valid_interval == 0:
                evaluation = evaluate(
                    crnn,
                    valid_loader,
                    criterion,
                    max_iter=args.valid_max_iter,
                    decode_method=args.decode_method,
                    beam_size=args.beam_size,
                )
                print("valid_evaluation: loss={loss}, acc={acc}".format(**evaluation))

                if i % args.save_interval == 0:
                    prefix = "crnn_synthetic_ocr_news"
                    loss_value = evaluation["loss"]
                    save_model_path = os.path.join(
                        args.checkpoints_dir, f"{prefix}_{i:06}_loss{loss_value}.pt"
                    )
                    torch.save(crnn.state_dict(), save_model_path)
                    print("save model at ", save_model_path)

            i += 1

        print("train_loss: ", tot_train_loss / tot_train_count)


if __name__ == "__main__":
    raise SystemExit(main())
