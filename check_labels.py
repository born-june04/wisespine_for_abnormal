import os
import numpy as np
import nibabel as nib
import multiprocessing

# Define paths
raw_labels_dir = "/gscratch/scrubbed/june0604/wisespine_for_abnormal/outputs/nnunet/nnUNet_raw/Dataset500_SpineAbnormal_Original/labelsTr"
preprocessed_labels_dir = "/gscratch/scrubbed/june0604/wisespine_for_abnormal/outputs/nnunet/nnUNet_preprocessed/Dataset500_SpineAbnormal_Original/gt_segmentations"

# Labels are 0 to 26
expected_labels = set(range(27))

def check_file(filepath):
    try:
        if filepath.endswith(".nii.gz"):
            img = nib.load(filepath)
            data = img.get_fdata()
        elif filepath.endswith(".npy"):
            data = np.load(filepath)
        elif filepath.endswith(".npz"):
            data = np.load(filepath)['data']
        else:
            return None

        unique_values = np.unique(data)
        invalid_values = [v for v in unique_values if int(v) not in expected_labels]
        
        if invalid_values:
            return f"\n[FAIL] {os.path.basename(filepath)}: Found invalid labels: {invalid_values}"
            
    except Exception as e:
        return f"\n[ERROR] {os.path.basename(filepath)}: Failed to read/process: {e}"
        
    return None

def get_files(directory):
    if not os.path.exists(directory):
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith((".nii.gz", ".npy", ".npz"))]

if __name__ == "__main__":
    raw_files = get_files(raw_labels_dir)
    preprocessed_files = get_files(preprocessed_labels_dir)
    
    all_files = raw_files + preprocessed_files
    print(f"Scanning {len(all_files)} files ({len(raw_files)} raw, {len(preprocessed_files)} preprocessed)...")
    print(f"Valid label range: 0-{max(expected_labels)}")
    
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.map(check_file, all_files)
        
    errors = [r for r in results if r is not None]
    
    if errors:
        print(f"\nFound {len(errors)} problematic files:")
        for e in errors:
            print(e)
    else:
        print("\n[SUCCESS] All label files contain valid values (0-26).")
