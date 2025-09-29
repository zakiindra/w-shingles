import os
import re
import zipfile
import shutil

# --- Configuration ---
# Set this to False only after you have reviewed the dry run output for Step 3.
DRY_RUN_RENAMING = True

# --- Paths ---
SOURCE_ZIPS_DIR = "article_dump"
EXTRACTED_CORPUS_DIR = "full_corpus_cleaned"

# --- Log Files ---
NAMING_ERROR_LOG = "naming_error.log"
MISSING_FILES_LOG = "missing_files.log"


def step_1_validate_zips(zip_dir):
    """
    Validates zip files for correct naming and expected file count.
    """
    print("\n--- Step 1: Validating Raw Zip Files ---")
    if os.path.exists(NAMING_ERROR_LOG): os.remove(NAMING_ERROR_LOG)
    if os.path.exists(MISSING_FILES_LOG): os.remove(MISSING_FILES_LOG)

    if not os.path.isdir(zip_dir):
        print(f"Error: Zip source directory not found at '{zip_dir}'")
        return False

    naming_pattern = re.compile(r'^[a-zA-Z\s._-]+_[A-Z]{2}\.zip$')
    naming_errors = False
    count_warnings = False
    
    for filename in os.listdir(zip_dir):
        if filename.endswith(".zip"):
            if not naming_pattern.match(filename):
                log_error(NAMING_ERROR_LOG, filename, f"Improper naming convention: '{filename}'")
                naming_errors = True
            
            try:
                with zipfile.ZipFile(os.path.join(zip_dir, filename), 'r') as zf:
                    file_list = [f for f in zf.namelist() if not f.startswith('__MACOSX/') and not f.endswith('/')]
                    if len(file_list) != 50:
                        log_error(MISSING_FILES_LOG, filename, f"Warning: Expected 50 files, but found {len(file_list)}.")
                        count_warnings = True
            except Exception as e:
                log_error(MISSING_FILES_LOG, filename, f"Could not be read: {e}")
                count_warnings = True
    
    if naming_errors:
        print(f"Warning: One or more zip files have an improper naming convention. See '{NAMING_ERROR_LOG}'.")
        print("  -> The pipeline will attempt to continue.")
        
    if count_warnings:
        print(f"Warning: One or more zip files do not contain 50 files. See '{MISSING_FILES_LOG}'.")
    
    if not naming_errors and not count_warnings:
        print("Validation successful.")

    return True


def step_2_extract_zips(zip_dir, output_dir):
    """
    Extracts all zip files, removes __MACOSX, and performs initial flattening.
    """
    print("\n--- Step 2: Extracting All Zip Archives ---")
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(zip_dir):
        if filename.endswith(".zip"):
            zip_path = os.path.join(zip_dir, filename)
            folder_name = os.path.splitext(filename)[0]
            city_dest_path = os.path.join(output_dir, folder_name)
            os.makedirs(city_dest_path, exist_ok=True)

            print(f"  Extracting '{filename}'...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    zf.extractall(city_dest_path)
            except Exception as e:
                print(f"    -> ERROR extracting: {e}")
                continue

            macosx_path = os.path.join(city_dest_path, "__MACOSX")
            if os.path.isdir(macosx_path): shutil.rmtree(macosx_path)

            nested_folder_path = os.path.join(city_dest_path, folder_name)
            if os.path.isdir(nested_folder_path):
                for item_name in os.listdir(nested_folder_path):
                    source_item = os.path.join(nested_folder_path, item_name)
                    dest_item = os.path.join(city_dest_path, item_name)
                    if os.path.exists(dest_item): os.remove(dest_item)
                    shutil.move(source_item, dest_item)
                os.rmdir(nested_folder_path)


def get_standardized_name(filename: str, dir_name: str) -> str:
    """
    Takes a messy filename and the correct directory name, and returns
    a standardized filename like 'City_ST_C-V.txt'.
    """
    match = re.search(r'(C(-\d+)?)(?:\.txt)?$', filename)
    if match:
        version_key = match.group(1)
        return f"{dir_name}_{version_key}.txt"
    else:
        return f"{dir_name}_C.txt"


def step_3_clean_and_standardize_files(corpus_dir):
    """
    Performs a multi-pass cleaning process for directories and files.
    """
    print("\n--- Step 3: Cleaning and Standardizing Extracted Files ---")
    if DRY_RUN_RENAMING:
        print("--- RUNNING IN DRY RUN MODE. NO FILES WILL BE MODIFIED. ---")
    else:
        print("--- RUNNING IN LIVE MODE. FILES WILL BE MODIFIED. ---")

    # Pass 1: Standardize directory names (e.g., 'Chicago_ IL' -> 'Chicago_IL')
    print("\n  -> Pass 1: Standardizing directory names...")
    dirs_to_process = [d.name for d in os.scandir(corpus_dir) if d.is_dir()]
    for dir_name in dirs_to_process:
        # A more direct fix for names like 'Chicago_ IL'
        new_dir_name = dir_name.replace(' _', '_')
        if " " in new_dir_name: # Also catches 'El Paso_TX' -> 'ElPaso_TX' if desired
             new_dir_name = new_dir_name.replace(' ', '')

        if dir_name != new_dir_name:
            print(f"    -> Rename Dir: '{dir_name}' >> '{new_dir_name}'")
            if not DRY_RUN_RENAMING:
                try:
                    os.rename(os.path.join(corpus_dir, dir_name), os.path.join(corpus_dir, new_dir_name))
                except Exception as e:
                    print(f"      ERROR: Could not rename directory. {e}")
    
    # Pass 2: Flatten any nested directories (like Chicago_IL/Chicago)
    print("\n  -> Pass 2: Flattening nested directories...")
    dirs_to_process = [d.name for d in os.scandir(corpus_dir) if d.is_dir()]
    for dir_name in dirs_to_process:
        dir_path = os.path.join(corpus_dir, dir_name)
        items_in_dir = os.listdir(dir_path)
        
        # This logic robustly finds any directory with a single subdirectory inside
        if len(items_in_dir) == 1 and os.path.isdir(os.path.join(dir_path, items_in_dir[0])):
            nested_dir_path = os.path.join(dir_path, items_in_dir[0])
            print(f"    -> Flattening: Moving files from '{nested_dir_path}'")
            if not DRY_RUN_RENAMING:
                for item_name in os.listdir(nested_dir_path):
                    shutil.move(os.path.join(nested_dir_path, item_name), dir_path)
                os.rmdir(nested_dir_path)

    # Pass 3: Standardize all filenames within the clean directories
    print("\n  -> Pass 3: Standardizing filenames...")
    dirs_to_process = [d.name for d in os.scandir(corpus_dir) if d.is_dir()]
    for dir_name in dirs_to_process:
        dir_path = os.path.join(corpus_dir, dir_name)
        for old_filename in os.listdir(dir_path):
            if not os.path.isfile(os.path.join(dir_path, old_filename)): continue # Skip directories
            new_filename = get_standardized_name(old_filename, dir_name)
            if old_filename != new_filename:
                print(f"    -> Rename File: '{old_filename}' >> '{new_filename}' in '{dir_name}'")
                if not DRY_RUN_RENAMING:
                    try:
                        os.rename(os.path.join(dir_path, old_filename), os.path.join(dir_path, new_filename))
                    except Exception as e:
                        print(f"      ERROR: Could not rename file. {e}")


def log_error(log_file, item_name, reason):
    with open(log_file, 'a') as f:
        f.write(f"{item_name}: {reason}\n")


def main():
    """Runs the full data preparation pipeline."""
    print("===== STARTING DATA PREPARATION PIPELINE =====")
    
    if step_1_validate_zips(SOURCE_ZIPS_DIR):
        step_2_extract_zips(SOURCE_ZIPS_DIR, EXTRACTED_CORPUS_DIR)
        step_3_clean_and_standardize_files(EXTRACTED_CORPUS_DIR)
    
    print("\n===== DATA PREPARATION PIPELINE COMPLETE =====")
    if DRY_RUN_RENAMING:
        print("NOTE: Renaming was in DRY RUN mode. Review the output, then set DRY_RUN_RENAMING = False and run again.")

if __name__ == "__main__":
    main()