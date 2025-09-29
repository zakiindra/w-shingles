import argparse
import glob
import hashlib
import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


def read_documents(dirpath: str) -> dict[str, str]:
    files = glob.glob(os.path.join(dirpath, "*_C*.txt"))
    documents = {}
    for filename in files:
        version = filename.replace(".txt", "").split("_")[-1]
        with open(filename, "r", encoding="utf-8") as f:
            documents[version] = f.read()

    return documents


def calculate_similarity(first: set[int], second: set[int]) -> float:
    if len(first) == 0 and len(second) == 0:
        return 1.0
    if len(second) == 0:
        return 0.0
    intersection = len(first.intersection(second))
    union = len(first.union(second))
    return intersection / union


def generate_fingerprint(document: str, window_size: int) -> set[int]:
    tokens = document.split(" ")
    raw_shingle = [] # for comparing unique shingles
    fingerprint = set()
    for i in range(0, len(tokens)):
        shingle = " ".join(tokens[i : i + window_size])
        raw_shingle.append(shingle)
        shingle_fingerprint = int(hashlib.md5(shingle.encode("UTF-8"), usedforsecurity=False).hexdigest(), 16)
        fingerprint.add(shingle_fingerprint)

    # print(len(raw_shingle), len(fingerprint))
    return fingerprint


def select_fingerprint(fingerprint: set[int], n: int) -> set[int]:
    if n == -1:
        return fingerprint
    return set(sorted(fingerprint)[:n])


def generate_fingerprints(documents: dict[str, str], w: int, n: int) -> dict[str, set[int]]:
    fingerprints = {}
    for version, text in documents.items():
        fingerprint = generate_fingerprint(text, w)
        fingerprints[version] = select_fingerprint(fingerprint, n)
    return fingerprints


def _offset_from_key(k: str) -> int:
    m = re.search(r"-\s*(\d+)$", k)
    return int(m.group(1)) if m else 0

def _choose_tick_indices(n: int, max_labels: int = 10) -> list[int]:
    """Pick up to `max_labels` evenly spaced indices from [0..n-1], always including first & last."""
    if n <= max_labels:
        return list(range(n))
    # even spacing, keep unique, ensure 0 and n-1 included
    idxs = np.linspace(0, n - 1, max_labels)
    idxs = sorted(set(int(round(i)) for i in idxs))
    # guard: if rounding caused duplicates, fill until we have <= max_labels unique
    while len(idxs) > max_labels:
        # remove a middle point
        mid = len(idxs) // 2
        del idxs[mid]
    if idxs[0] != 0:
        idxs[0] = 0
    if idxs[-1] != n - 1:
        idxs[-1] = n - 1
    return idxs

def plot_similarity(similarity: dict[str, float],
                    title: str = "Similarity vs. Version",
                    outfile: str = "similarity_trend.png",
                    max_xlabels: int = 10):
    if not similarity:
        raise ValueError("similarity dict is empty")

    # Sort by numeric offset (C-3, C-6, ...)
    items = sorted(similarity.items(), key=lambda kv: _offset_from_key(kv[0]))
    labels = [k for k, _ in items]
    values = [v for _, v in items]
    x = list(range(len(values)))

    plt.figure(figsize=(9, 4.8))
    # plot ALL data points, but no markers (no dots)
    plt.plot(x, values, linewidth=2)

    # Choose up to 10 tick positions, but keep all data plotted
    tick_idxs = _choose_tick_indices(len(x), max_labels=max_xlabels)
    tick_labels = [labels[i] for i in tick_idxs]
    plt.xticks(tick_idxs, tick_labels, ha="right")

    plt.xlabel("Version")
    plt.ylabel("Similarity")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.title(title)
    plt.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile, dpi=200)
    plt.close()
    print(f"Saved plot → {outfile}")

def main():
    argparser = argparse.ArgumentParser(description="W-shingling")
    argparser.add_argument("-d", "--dir", type=str, required=True, dest="dir", help="Directory containing input files")
    argparser.add_argument("-w", "--wsize", type=int, required=True, dest="w", help="Window size for w-shingling")
    argparser.add_argument("-n", "--lambda", type=int, required=True, dest="n", help="Lambda for w-shingling")
    argparser.add_argument("-o", "--outdir", type=str, required=True, dest="outdir", help="Output directory")
    args = argparser.parse_args()

    documents = read_documents(args.dir)
    fingerprints = generate_fingerprints(documents, args.w, args.n)

    first_fingerprint = fingerprints["C"]
    similarities = {}
    for version, fingerprint in fingerprints.items():
        similarity = calculate_similarity(first_fingerprint, fingerprints[version])
        similarities[version] = similarity

    os.makedirs(args.outdir, exist_ok=True)
    city = os.path.basename(args.dir)
    lam = str(args.n) if args.n != -1 else "all"
    outfile = os.path.join(args.outdir, f"{city}_{args.w}_{lam}.png")
    plot_similarity(similarities, title=f"Similarity {city} ({args.w}, {lam})", outfile=outfile)
    # print(sorted(similarities.items()))

    for w in [25, 50]:
        for n in [8, 16, 32, 64]:
            documents = read_documents(args.dir)
            fingerprints = generate_fingerprints(documents, w, n)

            first_fingerprint = fingerprints["C"]
            similarities = {}
            for version, fingerprint in fingerprints.items():
                similarity = calculate_similarity(first_fingerprint, fingerprints[version])
                similarities[version] = similarity

            os.makedirs(args.outdir, exist_ok=True)
            city = os.path.basename(args.dir)
            lam = str(n) if n != -1 else "all"
            outfile = os.path.join(args.outdir, f"{city}_{w}_{lam}.png")
            plot_similarity(similarities, title=f"Similarity {city} ({w}, {lam})", outfile=outfile)


if __name__ == "__main__":
    main()