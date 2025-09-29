import os
import glob
import hashlib
import time
import numpy as np
from multiprocessing import Pool, cpu_count
import re

# --- Configuration ---
CORPUS_DIR = "full_corpus_cleaned"
RESULTS_DIR = "results"
LAMBDA_VALS = [8, 16, 32, 64, -1]
W_VALS = [25, 50]
BATCH_SIZE = 10

# Logs
CITY_TIMING_LOG = os.path.join(RESULTS_DIR, "city_timings.log")
EXPLICIT_TIMING_LOG = os.path.join(RESULTS_DIR, "explicit_combo_timings.log")
SKIPPED_LOG_FILE = os.path.join(RESULTS_DIR, "skipped_cities.log")

# --- Core Functions ---
def calculate_similarity(first: set, second: set) -> float:
    if not first and not second:
        return 1.0
    if not first or not second:
        return 0.0
    return len(first.intersection(second)) / len(first.union(second))

def generate_fingerprint(text: str, w: int) -> set:
    tokens = text.split()
    fingerprint = set()
    if len(tokens) < w:
        return fingerprint
    for i in range(len(tokens) - w + 1):
        shingle = " ".join(tokens[i:i + w])
        hashed_shingle = int(hashlib.md5(shingle.encode("UTF-8")).hexdigest(), 16)
        fingerprint.add(hashed_shingle)
    return fingerprint

def select_fingerprint(fingerprint: set, n: int) -> set:
    if n == -1:
        return fingerprint
    return set(sorted(list(fingerprint))[:n])

def read_documents(dir_path: str) -> dict:
    docs = {}
    search_path = dir_path
    file_paths = glob.glob(os.path.join(search_path, "*"))

    if not any(os.path.isfile(p) for p in file_paths):
        dir_base_name = os.path.basename(dir_path).lower()
        for item in os.scandir(dir_path):
            if item.is_dir() and item.name.lower() == dir_base_name:
                search_path = item.path
                file_paths = glob.glob(os.path.join(search_path, "*"))
                break

    for file_path in file_paths:
        if os.path.isfile(file_path):
            filename = os.path.basename(file_path)
            match = re.search(r'(C(-\d+)?)(?:\.txt)?$', filename)
            if match:
                version_key = match.group(1)
                with open(file_path, "r", encoding="utf-8") as f:
                    docs[version_key] = f.read()
    return docs

# --- Worker Function for Per-City Totals ---
def process_city(args):
    total_start_time = time.perf_counter()
    city_dir, w_vals, full_fingerprints_cache = args
    city_name = os.path.basename(city_dir)
    city_docs = read_documents(city_dir)

    current_key = None
    if 'C' in city_docs:
        current_key = 'C'
    elif 'C-0' in city_docs:
        current_key = 'C-0'

    if current_key is None:
        return (city_name, None, 0.0, 0.0)

    similarity_time_accumulator = 0.0

    for w in w_vals:
        current_doc_id = (city_name, current_key)
        current_fp_full = full_fingerprints_cache[w][current_doc_id]
        current_fps = {lam: select_fingerprint(current_fp_full, lam) for lam in LAMBDA_VALS}
        for lam in LAMBDA_VALS:
            for version, text in city_docs.items():
                if version == current_key:
                    continue
                doc_id = (city_name, version)
                past_fp_full = full_fingerprints_cache[w][doc_id]
                past_fp_selected = select_fingerprint(past_fp_full, lam)

                sim_start = time.perf_counter()
                _ = calculate_similarity(current_fps[lam], past_fp_selected)
                sim_end = time.perf_counter()
                similarity_time_accumulator += (sim_end - sim_start)

    total_end_time = time.perf_counter()
    total_elapsed_time = total_end_time - total_start_time
    return (city_name, True, total_elapsed_time, similarity_time_accumulator)

# --- Explicit Combo Benchmarking ---
def benchmark_explicit_city(city_dir, full_fingerprints_cache):
    city_name = os.path.basename(city_dir)
    city_docs = read_documents(city_dir)

    current_key = None
    if 'C' in city_docs:
        current_key = 'C'
    elif 'C-0' in city_docs:
        current_key = 'C-0'
    if current_key is None:
        return []

    results = []
    for w in W_VALS:
        current_doc_id = (city_name, current_key)
        current_fp_full = full_fingerprints_cache[w][current_doc_id]

        for lam in LAMBDA_VALS:
            current_fp = select_fingerprint(current_fp_full, lam)
            total_times = []
            sim_times = []

            # warmup 3 times
            for _ in range(3):
                for version, text in city_docs.items():
                    if version == current_key:
                        continue
                    doc_id = (city_name, version)
                    past_fp_full = full_fingerprints_cache[w][doc_id]
                    past_fp = select_fingerprint(past_fp_full, lam)
                    _ = calculate_similarity(current_fp, past_fp)

            # measured runs (5 times)
            for _ in range(5):
                start = time.perf_counter()
                sim_acc = 0.0
                for version, text in city_docs.items():
                    if version == current_key:
                        continue
                    doc_id = (city_name, version)
                    past_fp_full = full_fingerprints_cache[w][doc_id]
                    past_fp = select_fingerprint(past_fp_full, lam)

                    sim_start = time.perf_counter()
                    _ = calculate_similarity(current_fp, past_fp)
                    sim_end = time.perf_counter()
                    sim_acc += (sim_end - sim_start)
                end = time.perf_counter()
                total_times.append(end - start)
                sim_times.append(sim_acc)

            results.append(
                (city_name, w, lam,
                 np.mean(total_times), np.std(total_times),
                 np.mean(sim_times), np.std(sim_times))
            )
    return results

# --- Main Runner ---
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    city_dirs = [d.path for d in os.scandir(CORPUS_DIR) if d.is_dir()]
    num_workers = cpu_count()

    all_results = {}
    skipped_cities = []
    city_total_timings = {}
    similarity_timings = {}
    explicit_results = []

    city_batches = [city_dirs[i:i + BATCH_SIZE] for i in range(0, len(city_dirs), BATCH_SIZE)]
    for batch in city_batches:
        batch_docs = {}
        for city_dir in batch:
            city_name = os.path.basename(city_dir)
            docs = read_documents(city_dir)
            for version, text in docs.items():
                batch_docs[(city_name, version)] = text

        batch_fingerprints_cache = {}
        for w in W_VALS:
            tasks = [(text, w) for text in batch_docs.values()]
            with Pool(processes=num_workers) as pool:
                results = pool.starmap(generate_fingerprint, tasks)
            doc_keys = list(batch_docs.keys())
            batch_fingerprints_cache[w] = {doc_keys[j]: results[j] for j in range(len(doc_keys))}

        tasks = [(city_dir, W_VALS, batch_fingerprints_cache) for city_dir in batch]
        with Pool(processes=num_workers) as pool:
            batch_results_list = pool.map(process_city, tasks)

        for result in batch_results_list:
            city, ok, total_time, sim_time = result
            if ok:
                all_results[city] = True
                city_total_timings[city] = total_time
                similarity_timings[city] = sim_time
                explicit_results.extend(
                    benchmark_explicit_city(city_dir=os.path.join(CORPUS_DIR, city),
                                            full_fingerprints_cache=batch_fingerprints_cache)
                )
            else:
                skipped_cities.append(city)

    # --- Write logs ---
    with open(CITY_TIMING_LOG, 'w', encoding='utf-8') as f:
        f.write("city_name,total_time,similarity_city,similarity_time\n")
        for city in sorted(city_total_timings.keys()):
            f.write(f"{city},{city_total_timings[city]:.4f},{city},{similarity_timings[city]:.4f}\n")

    with open(EXPLICIT_TIMING_LOG, 'w', encoding='utf-8') as f:
        f.write("city_name,w,lambda,mean_total_time,std_total_time,mean_similarity_time,std_similarity_time\n")
        for row in explicit_results:
            f.write(f"{row[0]},{row[1]},{row[2]},{row[3]:.4f},{row[4]:.4f},{row[5]:.4f},{row[6]:.4f}\n")

    if skipped_cities:
        with open(SKIPPED_LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("Skipped cities:\n")
            for city in skipped_cities:
                f.write(city + "\n")

    print(f"Logs saved:\n- {CITY_TIMING_LOG}\n- {EXPLICIT_TIMING_LOG}")

if __name__ == "__main__":
    if not os.path.isdir(CORPUS_DIR):
        print(f"Error: Base directory not found at '{CORPUS_DIR}'")
    else:
        main()
