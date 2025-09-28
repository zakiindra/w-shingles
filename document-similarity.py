import argparse
import glob
import hashlib
import os


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


def main():
    argparser = argparse.ArgumentParser(description="W-shingling")
    argparser.add_argument("-d", "--dir", required=True, dest="dir", help="Directory containing input files")
    argparser.add_argument("-w", "--wsize", type=int, required=True, dest="w", help="Window size for w-shingling")
    argparser.add_argument("-n", "--lambda", type=int, required=True, dest="n", help="Lambda for w-shingling")
    args = argparser.parse_args()

    documents = read_documents(args.dir)
    fingerprints = generate_fingerprints(documents, args.w, args.n)
    # for version, fingerprint in fingerprints.items():
    #     print(version, fingerprint)

    first_fingerprint = fingerprints["C"]
    similarities = {}
    for version, fingerprint in fingerprints.items():
        if version == "C":
            continue

        similarity = calculate_similarity(first_fingerprint, fingerprints[version])
        similarities[version] = similarity

    print(sorted(similarities.items()))


if __name__ == "__main__":
    main()