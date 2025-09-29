import os
import glob
import hashlib
import time
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Pool, cpu_count
import re
# --- Configuration ---
CORPUS_DIR = "full_corpus_cleaned"
RESULTS_DIR = "results"
LAMBDA_VALS = [8, 16, 32, 64, -1] # Use -1 for infinity case
W_VALS = [25, 50]
BATCH_SIZE = 10 
SKIPPED_LOG_FILE = os.path.join(RESULTS_DIR, "skipped_cities.log")

# --- Core Functions ---
def calculate_similarity(first: set, second: set) -> float:
    if not first and not second: return 1.0
    if not first or not second: return 0.0
    return len(first.intersection(second)) / len(first.union(second))

def generate_fingerprint(text: str, w: int) -> set:
    tokens = text.split()
    fingerprint = set()
    if len(tokens) < w: return fingerprint
    for i in range(len(tokens) - w + 1):
        shingle = " ".join(tokens[i:i + w])
        hashed_shingle = int(hashlib.md5(shingle.encode("UTF-8")).hexdigest(), 16)
        fingerprint.add(hashed_shingle)
    return fingerprint

def select_fingerprint(fingerprint: set, n: int) -> set:
    if n == -1: return fingerprint
    return set(sorted(list(fingerprint))[:n])

# --- MODIFIED Data Loading Function ---
def read_documents(dir_path: str) -> dict:
    docs = {}
    search_path = dir_path
    
    # First, try to find files in the main directory
    file_paths = glob.glob(os.path.join(search_path, "*"))

    # If it's empty, check for a nested directory (Edge Case 3)
    if not file_paths:
        dir_base_name = os.path.basename(dir_path).lower()
        for item in os.scandir(dir_path):
            if item.is_dir() and item.name.lower() == dir_base_name:
                print(f"  -> Found nested directory, searching in: {item.path}")
                search_path = item.path
                file_paths = glob.glob(os.path.join(search_path, "*"))
                break # Assume only one such nested folder exists

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        # Use regex to robustly find the version part
        match = re.search(r'(C(-\d+)?)(?:\.txt)?$', filename)
        if match:
            version_key = match.group(1)
            with open(file_path, "r", encoding="utf-8") as f:
                docs[version_key] = f.read()
    return docs

# --- Plotting Functions (No Changes) ---
def plot_combined_similarity(results: dict, city: str, w: int, output_dir: str):
    plt.figure(figsize=(12, 7))
    for lam_val, sim_data in results.items():
        lam_label = '∞' if lam_val == -1 else str(lam_val)
        versions = sorted(sim_data.keys(), key=lambda v: int(v.split('-')[-1]) if '-' in v else -1)
        sim_values = [sim_data[v] for v in versions]
        x_labels = [v.replace('C-', '') for v in versions]
        plt.plot(x_labels, sim_values, marker='o', linestyle='-', label=f'λ = {lam_label}')
    plt.title(f'Similarity Evolution for {city} (w={w})')
    plt.xlabel("Version (T - k)"); plt.ylabel("Jaccard Similarity with Current Version (T)")
    plt.grid(True); plt.legend(); plt.xticks(rotation=45); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"similarity_{city}_w{w}.png")); plt.close()

def plot_timing_graph(timings: dict, output_dir: str):
    plt.figure(figsize=(10, 6)); w_labels = [f"w = {w}" for w in timings.keys()]
    time_values = list(timings.values()); plt.bar(w_labels, time_values)
    plt.title('Total Time to Generate Full Shingle Sets for Entire Corpus')
    plt.ylabel("Time (seconds)"); plt.grid(axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shingle_generation_timing.png")); plt.close()
    
# --- MODIFIED Worker Function ---
def process_city(args):
    city_dir, w_vals, full_fingerprints_cache = args
    city_name = os.path.basename(city_dir)
    print(f"  Processing city: {city_name}...")
    city_docs = read_documents(city_dir)
    
    # Intelligently find the current version key (Edge Cases 1 & 2)
    current_key = None
    if 'C' in city_docs:
        current_key = 'C'
    elif 'C-0' in city_docs:
        current_key = 'C-0'
    
    if current_key is None: 
        print(f"  -> Skipping {city_name}: Current version ('_C.txt' or '_C-0.txt') not found.")
        return (city_name, None)

    city_results = {}
    for w in w_vals:
        city_results[w] = {}
        current_doc_id = (city_name, current_key)
        current_fp_full = full_fingerprints_cache[w][current_doc_id]
        current_fps = {lam: select_fingerprint(current_fp_full, lam) for lam in LAMBDA_VALS}
        for lam in LAMBDA_VALS:
            city_results[w][lam] = {}
            for version, text in city_docs.items():
                if version == current_key: continue
                doc_id = (city_name, version)
                past_fp_full = full_fingerprints_cache[w][doc_id]
                past_fp_selected = select_fingerprint(past_fp_full, lam)
                sim = calculate_similarity(current_fps[lam], past_fp_selected)
                city_results[w][lam][version] = sim
        plot_combined_similarity(city_results[w], city_name, w, RESULTS_DIR)
        
    return (city_name, city_results)

# --- Main Experiment Runner (No other changes needed) ---
def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    city_dirs = [d.path for d in os.scandir(CORPUS_DIR) if d.is_dir()]
    num_workers = cpu_count()
    print(f"Using {num_workers} CPU cores for parallel processing.")
    
    city_batches = [city_dirs[i:i + BATCH_SIZE] for i in range(0, len(city_dirs), BATCH_SIZE)]
    
    total_timings = {w: 0.0 for w in W_VALS}
    all_results = {}
    skipped_cities = [] 

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
            start_time = time.perf_counter()
            tasks = [(text, w) for text in batch_docs.values()]
            with Pool(processes=num_workers) as pool:
                results = pool.starmap(generate_fingerprint, tasks)
            
            doc_keys = list(batch_docs.keys())
            batch_fingerprints_cache[w] = {doc_keys[j]: results[j] for j in range(len(doc_keys))}
            end_time = time.perf_counter()
            total_timings[w] += (end_time - start_time)
            print(f"  Fingerprint generation for w={w} took {end_time - start_time:.4f} seconds.")

        tasks = [(city_dir, W_VALS, batch_fingerprints_cache) for city_dir in batch]
        with Pool(processes=num_workers) as pool:
            batch_results_list = pool.map(process_city, tasks)
        
        for result in batch_results_list:
            city, data = result
            if data is not None:
                all_results[city] = data
            else:
                skipped_cities.append(city)

    print("\nStep 4: Analyzing final results...")
    for w in W_VALS:
        errors = {lam: [] for lam in LAMBDA_VALS if lam != -1}
        for city_data in all_results.values():
            if w in city_data and -1 in city_data[w]:
                for version in city_data[w][-1]:
                    inf_sim = city_data[w][-1][version]
                    for lam in errors.keys():
                        lam_sim = city_data[w][lam][version]
                        errors[lam].append(abs(inf_sim - lam_sim))
        
        avg_errors = {lam: np.mean(errs) if errs else 0 for lam, errs in errors.items()}
        if any(avg_errors.values()):
            best_lam = min(avg_errors, key=avg_errors.get)
            print(f"  For w={w}, the λ value closest to ∞ is {best_lam} (Avg. Difference: {avg_errors[best_lam]:.6f})")

    plot_timing_graph(total_timings, RESULTS_DIR)

    if skipped_cities:
        print(f"\nWarning: {len(skipped_cities)} cities were skipped. See '{SKIPPED_LOG_FILE}' for details.")
        with open(SKIPPED_LOG_FILE, 'w', encoding='utf-8') as f:
            f.write("The following cities were skipped because a current version ('_C.txt' or '_C-0.txt') was not found:\n")
            for city_name in sorted(skipped_cities):
                f.write(f"{city_name}\n")

    print(f"\n--- Experiments Complete ---")
    print(f"Total shingle generation time for w=25: {total_timings[25]:.4f}s")
    print(f"Total shingle generation time for w=50: {total_timings[50]:.4f}s")
    print(f"All result graphs have been saved in the '{RESULTS_DIR}' directory.")

if __name__ == "__main__":
    if not os.path.isdir(CORPUS_DIR):
        print(f"Error: Base directory not found at '{CORPUS_DIR}'")
    else:
        main()
