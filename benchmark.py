import os
import glob
import hashlib
import time
import numpy as np
from multiprocessing import Pool, cpu_count
import re

CORPUS_DIR = "full_corpus_cleaned"
RESULTS_DIR = "logs"

LAMBDA_VALS = [8, 16, 32, 64, -1]
W_VALS = [25, 50]
BATCH_SIZE = 10
# Set the number of workers, leaving some cores for system overhead
NUM_WORKERS = 60

os.mkdir(RESULTS_DIR, exist_ok=True)
# Logs
CITY_TIMING_LOG = os.path.join(RESULTS_DIR, "city_timings.log")
SHINGLES_AND_LAMBDA_LOG = os.path.join(RESULTS_DIR, "shingles_and_lambda_timing.log")
SKIPPED_LOG_FILE = os.path.join(RESULTS_DIR, "skipped_cities_benchmark.log")

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

def benchmark_city_for_w(args):
    city_dir, w, full_fingerprints_cache = args
    city_name = os.path.basename(city_dir)
    city_docs = read_documents(city_dir)

    current_key = None
    if 'C' in city_docs:
        current_key = 'C'
    elif 'C-0' in city_docs:
        current_key = 'C-0'

    if current_key is None:
        return (city_name, w, None, None, None)

    total_start_time = time.perf_counter()
    similarity_time_accumulator = 0.0
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
    total_elapsed_time = time.perf_counter() - total_start_time

    explicit_results_for_w = []
    for lam in LAMBDA_VALS:
        current_fp = select_fingerprint(current_fp_full, lam)
        total_times, sim_times = [], []

        # Warmup runs
        for _ in range(3):
            for version, text in city_docs.items():
                if version == current_key: continue
                past_fp = select_fingerprint(full_fingerprints_cache[w][(city_name, version)], lam)
                _ = calculate_similarity(current_fp, past_fp)

        # Measured runs
        for _ in range(5):
            start = time.perf_counter()
            sim_acc = 0.0
            for version, text in city_docs.items():
                if version == current_key: continue
                past_fp = select_fingerprint(full_fingerprints_cache[w][(city_name, version)], lam)
                sim_start = time.perf_counter()
                _ = calculate_similarity(current_fp, past_fp)
                sim_end = time.perf_counter()
                sim_acc += (sim_end - sim_start)
            end = time.perf_counter()
            total_times.append(end - start)
            sim_times.append(sim_acc)

        explicit_results_for_w.append(
            (city_name, w, lam,
             np.mean(total_times), np.std(total_times),
             np.mean(sim_times), np.std(sim_times))
        )
    
    return (city_name, w, total_elapsed_time, similarity_time_accumulator, explicit_results_for_w)

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    city_dirs = [d.path for d in os.scandir(CORPUS_DIR) if d.is_dir()]
    num_workers = min(NUM_WORKERS, cpu_count())
    print(f"Using {num_workers} CPU cores for parallel processing.")

    skipped_cities = []
    city_total_timings = {}
    similarity_timings = {}
    explicit_results = []

    city_batches = [city_dirs[i:i + BATCH_SIZE] for i in range(0, len(city_dirs), BATCH_SIZE)]
    for i, batch in enumerate(city_batches):
        print(f"\n--- Processing Batch {i + 1}/{len(city_batches)} ---")
        
        batch_docs = {}
        for city_dir in batch:
            city_name = os.path.basename(city_dir)
            docs = read_documents(city_dir)
            for version, text in docs.items():
                batch_docs[(city_name, version)] = text

        batch_fingerprints_cache = {}
        for w in W_VALS:
            print(f"  Generating fingerprints for w={w}...")
            tasks = [(text, w) for text in batch_docs.values()]
            with Pool(processes=num_workers) as pool:
                results = pool.starmap(generate_fingerprint, tasks)
            doc_keys = list(batch_docs.keys())
            batch_fingerprints_cache[w] = {doc_keys[j]: results[j] for j in range(len(doc_keys))}

        print(f"  Benchmarking similarities for {len(batch)} cities...")
        tasks = [(city_dir, w, batch_fingerprints_cache) for city_dir in batch for w in W_VALS]
        with Pool(processes=num_workers) as pool:
            batch_results_list = pool.map(benchmark_city_for_w, tasks)

        for result in batch_results_list:
            city, w, total_time, sim_time, explicit_res = result
            if total_time is not None:
                city_total_timings[city] = city_total_timings.get(city, 0) + total_time
                similarity_timings[city] = similarity_timings.get(city, 0) + sim_time
                explicit_results.extend(explicit_res)
            else:
                if city not in skipped_cities:
                    print(f"  -> Skipping {city}: Current version not found.")
                    skipped_cities.append(city)

    print("\n--- Writing Log Files ---")
    with open(CITY_TIMING_LOG, 'w', encoding='utf-8') as f:
        f.write("city_name,total_time_all_w,similarity_time_all_w\n")
        for city in sorted(city_total_timings.keys()):
            f.write(f"{city},{city_total_timings[city]:.4f},{similarity_timings[city]:.4f}\n")

    with open(SHINGLES_AND_LAMBDA_LOG, 'w', encoding='utf-8') as f:
        f.write("city_name,w,lambda,mean_total_time,std_total_time,mean_similarity_time,std_similarity_time\n")
        for row in explicit_results:
            lam_str = 'inf' if row[2] == -1 else row[2]
            f.write(f"{row[0]},{row[1]},{lam_str},{row[3]:.4f},{row[4]:.4f},{row[5]:.4f},{row[6]:.4f}\n")

    if skipped_cities:
        with open(SKIPPED_LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("Skipped cities (current version not found):\n")
            for city in sorted(skipped_cities):
                f.write(city + "\n")

    print(f"Logs saved:\n- {CITY_TIMING_LOG}\n- {SHINGLES_AND_LAMBDA_LOG}")
    if skipped_cities:
        print(f"- {SKIPPED_LOG_FILE}")

if __name__ == "__main__":
    if not os.path.isdir(CORPUS_DIR):
        print(f"Error: Base directory not found at '{CORPUS_DIR}'")
    else:
        main()