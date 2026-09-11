import random
import sys
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    import ants
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from preprocess import Preprocess
    from artifact_segmentation.anchor import AxialAnchor
    from artifact_segmentation.tracker import Tracker
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install antspyx torch scikit-image")
    sys.exit(1)


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """Global configuration"""
    
    # Paths (will be updated via command line)
    DATA_DIR = Path('../data/Train')
    OUTPUT_DIR = Path('../data')
    TEMPLATE_PATH = '../templates/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii'
    
    # Subject processing
    N_SUBJECTS = 200
    SUBJECT_PREFIX = 'subj'  # e.g., subj001, subj002, ...
    
    # Processing options
    SKIP_EXISTING = True
    
    # Dataset splits
    TRAIN_RATIO = 0.80
    VAL_RATIO = 0.2
    TEST_RATIO = 0

def process_single_subject(subject_id, config):     
    try:
        # Load data
        subject_dir = config.DATA_DIR / f'{config.SUBJECT_PREFIX}{subject_id}'
        img_path = subject_dir / f'subj{subject_id}_image.nii.gz'  
        mask_path = subject_dir / f'subj{subject_id}_artifact.nii.gz'
        
        if not img_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Files not found for {subject_id}")
        
        original_img = ants.image_read(str(img_path))
        original_mask = ants.image_read(str(mask_path))
        
        img_array = original_img.numpy()
        mask_array = original_mask.numpy().astype(bool)
        predicted_mask = np.zeros_like(mask_array)
        
        
        preprocess = Preprocess(config.TEMPLATE_PATH)
        reg_image, reg_mask, template = preprocess.preprocess_subject(str(img_path), str(mask_path), subject_dir)

        img_data = reg_image.numpy()

        # Find anchor
        axial_anchor = AxialAnchor()
        axial_anchor_idx, axial_anchor_data = axial_anchor.find_anchor(
            img_data=img_data,
            target_slices=range(0, img_data.shape[2]),
            axis=2,
            verbose=False
        )

        tracker = Tracker(search_radius=10)
        artifact_combined, tracking_info = tracker.track_artifact(img_data, axial_anchor_idx, axial_anchor_data['artifact_mask'])

        predicted_mask = tracker.smooth_artifact(artifact_combined)

        pred_img = reg_image.new_image_like(predicted_mask.astype(float))
        ants.image_write(pred_img, str(subject_dir / f'subj{subject_id}_artifact_estimation.nii.gz')) 
    
    except Exception as e:
        raise e

def batch_process_all(config):    
    print("="*80)
    print("BATCH PROCESSING - 200 SUBJECTS")
    print("="*80)
    
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    subject_ids = [f'{i:03d}' for i in range(1, config.N_SUBJECTS + 1)]
    
    for subject_id in tqdm(subject_ids, desc="Processing subjects"):
        
        # Skip if already done
        if config.SKIP_EXISTING:
            subject_dir = config.DATA_DIR / f'{config.SUBJECT_PREFIX}{subject_id}'
            estimation_file = subject_dir / f'{subject_id}_artifact_estimation.nii.gz'
            if estimation_file.exists():
                continue
        
        # Process
        process_single_subject(subject_id, config)

def create_unet_dataset(config, seed=42):

    random.seed(seed)

    dataset_root = (
        config.OUTPUT_DIR /
        "unet_dataset"
    )

    # create folders
    for split in ['train', 'val']:
        (
            dataset_root / split
        ).mkdir(
            parents=True,
            exist_ok=True
        )

    # subject ids
    subject_ids = [
        f'{i:03d}'
        for i in range(
            1,
            config.N_SUBJECTS + 1
        )
    ]

    random.shuffle(subject_ids)

    n = len(subject_ids)

    train_end = int(config.TRAIN_RATIO * n)
    val_end = train_end + int(config.VAL_RATIO * n)

    split_map = {
        'train': subject_ids[:train_end],
        'val': subject_ids[train_end:val_end],
    }

    print("=" * 80)
    print("CREATING U-NET DATASET")
    print("=" * 80)

    saved_count = 0

    for split, ids in split_map.items():

        print(f"\n{split.upper()} ({len(ids)} subjects)")

        for subject_id in tqdm(ids):

            try:

                subject_dir = (
                    config.DATA_DIR /
                    f'{config.SUBJECT_PREFIX}{subject_id}'
                )

                img_path = (
                    subject_dir /
                    f'subj{subject_id}_image.nii.gz'
                )

                gt_path = (
                    subject_dir /
                    f'subj{subject_id}_artifact.nii.gz'
                )

                est_path = (
                    subject_dir /
                    f'subj{subject_id}_artifact_estimation.nii.gz'
                )

                # skip missing
                if not (
                    img_path.exists()
                    and gt_path.exists()
                    and est_path.exists()
                ):
                    print(
                        f"Skipping {subject_id}"
                    )
                    continue

                # load
                img = ants.image_read(
                    str(img_path)
                )

                gt = ants.image_read(
                    str(gt_path)
                )

                est = ants.image_read(
                    str(est_path)
                )

                img_np = img.numpy().astype(np.float32)
                est_np = est.numpy().astype(np.float32)
                gt_np = gt.numpy().astype(np.float32)

                # sanity check
                if (
                    img_np.shape != est_np.shape
                    or img_np.shape != gt_np.shape
                ):
                    print(
                        f"Shape mismatch: {subject_id}"
                    )
                    continue

                # stack channels
                combined = np.stack(
                    [
                        img_np,
                        est_np,
                        gt_np
                    ],
                    axis=-1
                )

                # create 4D nifti
                combined_img = ants.from_numpy(
                    combined,
                    spacing=img.spacing,
                    origin=img.origin,
                    direction=img.direction
                )

                save_path = (
                    dataset_root /
                    split /
                    f'{subject_id}.nii.gz'
                )

                ants.image_write(
                    combined_img,
                    str(save_path)
                )

                saved_count += 1

            except Exception as e:

                print(
                    f"Failed {subject_id}: {e}"
                )

    print("\n" + "=" * 80)
    print("DATASET COMPLETE")
    print("=" * 80)

    print(
        f"Saved {saved_count} subjects"
    )

    print(
        f"Location:\n{dataset_root}"
    )

    return dataset_root

if __name__ == '__main__':
    config = Config()
    batch_process_all(config)
    create_unet_dataset(config, seed=42)
    