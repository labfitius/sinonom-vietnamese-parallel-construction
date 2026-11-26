"""5-fold page-grouped ConvNeXt-Tiny classifier on teacher's high-confidence (Correctness=1) crops.

Zero page leakage: all crops from the same page stay in the same fold.
For every crop (including Correctness=0 errors!), saves out-of-fold:
  - top-30 predicted chars & probabilities
  - probabilities for ALL dictionary chars matching the aligned Vietnamese syllable (exact + base-fuzzy)
Also trains/saves checkpoints in models/convnext_nom/ for reuse on the remaining 20 books in Task 1.

Usage:
  python scripts/opt_train_classifier.py --folds 0 1 2 3 4 --epochs 18
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8")
from core.nomdict import NomDict, syl_key, key_base  # noqa: E402

WORK = ROOT / "optimize" / "_work"
MODELS_DIR = ROOT / "models" / "convnext_nom"
BOOKS = [
    "Ca Trù Thể Cách, Nôm",
    "Nhi thap tu nu tac dien am digcoll_12920",
    "Thánh Du Quảng Huân Diên Nghĩa, Nôm",
    "Tây Du Truyên, Nôm",
]


def load_all_crops():
    imgs, labels, syls, corrs, pages, lines, poss, books, cands = [], [], [], [], [], [], [], [], []
    for bi, b in enumerate(BOOKS):
        d = np.load(WORK / f"crops_{b}.npz", allow_pickle=True)
        n = len(d["label"])
        imgs.append(d["img"])
        labels.append(d["label"])
        syls.append(d["syl"])
        corrs.append(d["corr"])
        pages.append(d["page"])
        lines.append(d["line"])
        poss.append(d["pos"])
        cands.append(d["cands"])
        books.append(np.full(n, b, dtype=object))
    return {
        "img": np.concatenate(imgs, axis=0),
        "label": np.concatenate(labels, axis=0),
        "syl": np.concatenate(syls, axis=0),
        "corr": np.concatenate(corrs, axis=0),
        "page": np.concatenate(pages, axis=0),
        "line": np.concatenate(lines, axis=0),
        "pos": np.concatenate(poss, axis=0),
        "cands": np.concatenate(cands, axis=0),
        "book": np.concatenate(books, axis=0),
    }


def make_page_folds(books, pages, n_folds=5, seed=42):
    """Stratified by book: assign each unique page deterministically to fold 0..n_folds-1."""
    rng = np.random.default_rng(seed)
    page_to_fold = {}
    for b in BOOKS:
        b_pages = sorted(set(pages[books == b]))
        perm = rng.permutation(len(b_pages))
        for rank, idx in enumerate(perm):
            page_to_fold[b_pages[idx]] = rank % n_folds
    return np.array([page_to_fold[p] for p in pages], dtype=np.int32)


class CropDS(Dataset):
    def __init__(self, imgs, y=None, train=False, img_size=96):
        self.imgs = imgs
        self.y = y
        self.train = train
        if train:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.RandomAffine(degrees=6, translate=(0.06, 0.06), scale=(0.88, 1.12), fill=255),
                transforms.ColorJitter(brightness=0.15, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        x = self.tf(self.imgs[i])
        x = x.repeat(3, 1, 1)  # 3 channels for pretrained ConvNeXt
        if self.y is None:
            return x, i
        return x, int(self.y[i])


def build_model(num_classes: int, arch: str = "convnext_tiny.fb_in22k_ft_in1k"):
    return timm.create_model(arch, pretrained=True, num_classes=num_classes, drop_path_rate=0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--img_size", type=int, default=96)
    ap.add_argument("--arch", type=str, default="convnext_tiny.fb_in22k_ft_in1k")
    ap.add_argument("--lr", type=float, default=4e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | arch: {args.arch} | img_size: {args.img_size}")

    data = load_all_crops()
    folds = make_page_folds(data["book"], data["page"], n_folds=5, seed=42)
    # Vocabulary: all unique characters with corr==1
    vocab = sorted(set(data["label"][data["corr"] == 1]))
    c2i = {c: i for i, c in enumerate(vocab)}
    print(f"Total crops: {len(data['label'])} | corr==1: {int(data['corr'].sum())} | vocab: {len(vocab)}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (MODELS_DIR / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")

    # Load dictionary v2 and also index by tone-stripped base for fuzzy fallback
    nd = NomDict.load()
    base_to_chars = {}
    for ch, ks in nd.c2k.items():
        if ch in c2i:
            for k in ks:
                base_to_chars.setdefault(key_base(k), set()).add(ch)
    # Also add teacher-observed (char, syl) pairs from TRAINING fold only inside each fold!

    for fold in args.folds:
        t0 = time.time()
        tr_mask = (folds != fold) & (data["corr"] == 1)
        va_mask = (folds == fold)
        va_ok_mask = va_mask & (data["corr"] == 1)

        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]
        y_tr = np.array([c2i[c] for c in data["label"][tr_idx]], dtype=np.int64)
        y_va_ok = np.array([c2i.get(c, -1) for c in data["label"][va_idx]], dtype=np.int64)

        tr_loader = DataLoader(
            CropDS(data["img"][tr_idx], y_tr, train=True, img_size=args.img_size),
            batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True,
        )
        va_loader = DataLoader(
            CropDS(data["img"][va_idx], None, train=False, img_size=args.img_size),
            batch_size=args.batch * 2, shuffle=False, num_workers=0, pin_memory=True,
        )

        # Build fold-specific dictionary (NomDict v2 + pairs seen in tr_idx only -> zero leakage!)
        fold_k2c = {k: set(v) for k, v in nd.k2c.items()}
        fold_b2c = {b: set(v) for b, v in base_to_chars.items()}
        for i in tr_idx:
            ch = data["label"][i]
            k = syl_key(data["syl"][i])
            fold_k2c.setdefault(k, set()).add(ch)
            fold_b2c.setdefault(key_base(k), set()).add(ch)

        model = build_model(len(vocab), args.arch).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        best_acc = 0.0
        best_state = None
        print(f"\n===== FOLD {fold} | train={len(tr_idx)} val={len(va_idx)} (corr=1: {int(va_ok_mask.sum())}) =====")
        for ep in range(1, args.epochs + 1):
            model.train()
            loss_sum = 0.0; n_seen = 0
            for xb, yb in tr_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(xb)
                    loss = crit(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                loss_sum += loss.item() * len(yb); n_seen += len(yb)
            sched.step()

            # Eval every 3 epochs and on last 4 epochs
            if ep % 3 == 0 or ep >= args.epochs - 3:
                model.eval()
                correct1 = correct5 = total_eval = 0
                with torch.no_grad():
                    for xb, loc_i in va_loader:
                        xb = xb.to(device, non_blocking=True)
                        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                            logits = model(xb)
                        top5 = logits.topk(5, dim=1).indices.cpu().numpy()
                        yt = y_va_ok[loc_i.numpy()]
                        valid = (data["corr"][va_idx[loc_i.numpy()]] == 1) & (yt >= 0)
                        if valid.any():
                            correct1 += int((top5[valid, 0] == yt[valid]).sum())
                            correct5 += int(np.any(top5[valid] == yt[valid, None], axis=1).sum())
                            total_eval += int(valid.sum())
                acc1 = correct1 / max(total_eval, 1)
                acc5 = correct5 / max(total_eval, 1)
                print(f"  ep {ep:02d}/{args.epochs} loss={loss_sum/n_seen:.3f} val_top1={acc1:.4f} val_top5={acc5:.4f} ({time.time()-t0:.0f}s)")
                if acc1 > best_acc:
                    best_acc = acc1
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        ckpt_path = MODELS_DIR / f"fold{fold}.pt"
        torch.save({"arch": args.arch, "img_size": args.img_size, "val_top1": best_acc, "state_dict": best_state}, ckpt_path)
        print(f"Saved {ckpt_path} (best val_top1={best_acc:.4f})")

        # OOF inference on all val crops (including corr==0!)
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        oof_records = []
        with torch.no_grad():
            for xb, loc_i in va_loader:
                xb = xb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    logits = model(xb).float()
                probs = F.softmax(logits, dim=1)
                top_p, top_idx = probs.topk(30, dim=1)
                top_p = top_p.cpu().numpy()
                top_idx = top_idx.cpu().numpy()
                probs_cpu = probs.cpu().numpy()
                for b_j, li in enumerate(loc_i.numpy()):
                    g_idx = int(va_idx[li])
                    syl = str(data["syl"][g_idx])
                    sk = syl_key(syl)
                    # Exact reading candidates in vocab
                    ex_chars = [c for c in fold_k2c.get(sk, ()) if c in c2i]
                    ex_scores = sorted(
                        [(c, float(probs_cpu[b_j, c2i[c]])) for c in ex_chars],
                        key=lambda x: -x[1],
                    )[:20]
                    # Fuzzy (base-only) candidates in vocab
                    fz_chars = [c for c in fold_b2c.get(key_base(sk), ()) if c in c2i and c not in ex_chars]
                    fz_scores = sorted(
                        [(c, float(probs_cpu[b_j, c2i[c]])) for c in fz_chars],
                        key=lambda x: -x[1],
                    )[:10]
                    oof_records.append({
                        "g_idx": g_idx,
                        "book": str(data["book"][g_idx]),
                        "page": str(data["page"][g_idx]),
                        "line": int(data["line"][g_idx]),
                        "pos": int(data["pos"][g_idx]),
                        "top_labels": [vocab[int(k)] for k in top_idx[b_j]],
                        "top_scores": [round(float(s), 6) for s in top_p[b_j]],
                        "dict_exact": [[c, round(s, 6)] for c, s in ex_scores],
                        "dict_fuzzy": [[c, round(s, 6)] for c, s in fz_scores],
                    })
        oof_path = WORK / f"oof_fold{fold}.jsonl"
        with open(oof_path, "w", encoding="utf-8") as f:
            for r in oof_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {oof_path} ({len(oof_records)} crops)")


if __name__ == "__main__":
    main()
