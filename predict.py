from preprocess import Preprocess
from artifact_segmentation.anchor import AxialAnchor
from artifact_segmentation.tracker import Tracker
from catheter_segmentation.unet import UNet3D
from catheter_length_estimation.estimate_length import run_inference


import numpy as np
import torch
import ants
from tqdm import tqdm
from pathlib import Path
from monai.networks.nets import UNet
import skimage.transform as skT
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

class Config:
    DATA_DIR = Path('./data/Train')
    OUTPUT_DIR = Path('./output')
    TEMPLATE_PATH = './templates/mni_icbm152_nlin_asym_09c/mni_icbm152_t1_tal_nlin_asym_09c.nii'
    
    # Subject processing
    N_SUBJECTS = 30
    SUBJECT_PREFIX = 'subj'  # e.g., subj001, subj002, ...

def process_single_subject(subject_id, config):     
    # Load data
    subject_dir = config.DATA_DIR / f'{config.SUBJECT_PREFIX}{subject_id}'
    subject_dir.mkdir(parents=True, exist_ok=True)

    img_path = config.DATA_DIR / 'subj010' /f'subj{subject_id}_image.nii.gz'
    
    if not img_path.exists():
        raise FileNotFoundError(f"Files not found for {subject_id}")
    
    original_img = ants.image_read(str(img_path))
    
    img_array = original_img.numpy()
    predicted_mask = np.zeros_like(img_array)
    
    preprocess = Preprocess(config.TEMPLATE_PATH)
    reg_image, reg_mask, template = preprocess.preprocess_subject(str(img_path), None, subject_dir)

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

    return original_img, reg_image, predicted_mask

def inverse_transform_mask(mask_array, original_img, reg_image, path):
    transforms = [
        str(path / 'reg_1InverseWarp.nii.gz'),  # Already inverse
        str(path / 'reg_0GenericAffine.mat')     # Needs inversion
    ]
    mask_ants = ants.from_numpy(mask_array.astype('float32'))
    ants.copy_image_info(reg_image, mask_ants)
    
    result = ants.apply_transforms(
        fixed=original_img,              # Target (original space)
        moving=mask_ants,                # Source (registered space)
        transformlist=transforms,
        whichtoinvert=[False, True],     # Don't invert InverseWarp, DO invert Affine
        interpolator='nearestNeighbor'
    )
    
    mask_original = result.numpy().astype(bool)
    
    return mask_original

def sliding_window_inference(model, volume, patch_size=64, stride=32, device='cuda', threshold=0.5):
    model.eval()
    D, H, W = volume.shape
    p = patch_size

    pred_sum = np.zeros((D, H, W), dtype=np.float32)
    count_map = np.zeros((D, H, W), dtype=np.float32)

    # Normalize volume
    vol_norm = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
    vol_tensor = torch.tensor(vol_norm).unsqueeze(0).unsqueeze(0).float()  # (1,1,D,H,W)

    with torch.no_grad():
        for z in range(0, D - p + 1, stride):
            for y in range(0, H - p + 1, stride):
                for x in range(0, W - p + 1, stride):
                    patch = vol_tensor[:, :, z:z+p, y:y+p, x:x+p].to(device)
                    logit = model(patch)
                    prob = torch.sigmoid(logit).squeeze().cpu().numpy()
                    pred_sum[z:z+p, y:y+p, x:x+p] += prob
                    count_map[z:z+p, y:y+p, x:x+p] += 1

    # Average overlapping predictions
    count_map = np.maximum(count_map, 1)
    avg_pred = pred_sum / count_map
    return (avg_pred > threshold).astype(np.uint8)

def segment_artifact(image, estimation, model, device):
    roi = np.where(
        estimation > 0.5
    )

    if len(roi[0]) > 0:

        margin = 30

        x_min = max(
            0,
            roi[0].min() - margin
        )

        x_max = min(
            estimation.shape[0]-1,
            roi[0].max() + margin
        )

        y_min = max(
            0,
            roi[1].min() - margin
        )

        y_max = min(
            estimation.shape[1]-1,
            roi[1].max() + margin
        )

        z_min = max(
            0,
            roi[2].min() - margin
        )

        z_max = min(
            estimation.shape[2]-1,
            roi[2].max() + margin
        )

    else:
        x_min, x_max = 0, estimation.shape[0]-1
        y_min, y_max = 0, estimation.shape[1]-1
        z_min, z_max = 0, estimation.shape[2]-1

    image_crop = image[
        x_min:x_max+1,
        y_min:y_max+1,
        z_min:z_max+1
    ]

    estimation_crop = estimation[
        x_min:x_max+1,
        y_min:y_max+1,
        z_min:z_max+1
    ]

    image_crop = skT.resize(
        image_crop,
        (96,96,96),
        order=1,
        preserve_range=True
    )

    estimation_crop = skT.resize(
        estimation_crop,
        (96,96,96),
        order=1,
        preserve_range=True
    )

    x = np.stack(
        [
            image_crop,
            estimation_crop
        ],
        axis=0
    ).astype(np.float32)

    x = torch.from_numpy(
        x
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        pred = model(x)

        pred = torch.sigmoid(
            pred
        )

        pred = (
            pred > 0.5
        ).float()

        pred = (
            pred.cpu()
            .numpy()[0,0]
        )

    pred = skT.resize(
        pred.astype(np.float32),
        image[
            x_min:x_max+1,
            y_min:y_max+1,
            z_min:z_max+1
        ].shape,
        order=0,
        preserve_range=True
    )

    pred = (
        pred > 0.5
    )


    pred_full = np.zeros(
        image.shape,
        dtype=np.uint8
    )

    pred_full[
        x_min:x_max+1,
        y_min:y_max+1,
        z_min:z_max+1
    ] = pred

    return pred_full

if __name__ == '__main__':   
    config = Config() 
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    catheter_segmentation_model = UNet3D(base_channels=16).to(device)
    checkpoint = torch.load('models/catheter_seg_model.pth', map_location=device)
    state_dict = checkpoint['model_state_dict']
    catheter_segmentation_model.load_state_dict(state_dict)
    catheter_segmentation_model.to(device)
    catheter_segmentation_model.eval()

    artifact_segmentation_model = model = UNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=1,
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2),
        num_res_units=2,
    ).to(device)
    state_dict = torch.load('models/artifact_seg_model.pth', map_location=device)
    checkpoint = checkpoint['model_state_dict']
    artifact_segmentation_model.load_state_dict(state_dict)
    artifact_segmentation_model.to(device)
    artifact_segmentation_model.eval()

    output_data = []
    
    subject_ids = [f'{i:03d}' for i in range(0 , 1)]
    
    for subject_id in tqdm(subject_ids, desc="Processing subjects"):
        

        subject_id = '010'
        subject_dir = config.DATA_DIR / f'{config.SUBJECT_PREFIX}{subject_id}'

        original_img, reg_img, estimated_artifact_mask = process_single_subject(subject_id, config)

        estimated_artifact_mask = estimated_artifact_mask.astype(float)

        reg_predicted_artifact_mask = segment_artifact(reg_img.numpy(), estimated_artifact_mask, artifact_segmentation_model, device)

        predicted_artifact_mask = inverse_transform_mask(
            reg_predicted_artifact_mask,
            original_img,
            reg_img,
            subject_dir
        )

        predicted_artifact_mask = original_img.new_image_like(predicted_artifact_mask.astype(float))
        ants.image_write(predicted_artifact_mask, str(config.OUTPUT_DIR / f'subj{subject_id}_taskA.nii.gz'))
        # predicted_artifact_mask = ants.image_read(str(config.OUTPUT_DIR / f'subj{subject_id}_taskA.nii.gz'))

        # predicted_catheter_mask = sliding_window_inference(catheter_segmentation_model, original_img.numpy(), patch_size=64, stride=32, device=device)

        # predicted_catheter_mask = ants.from_numpy(predicted_catheter_mask.astype(np.uint8))
        # predicted_catheter_mask = ants.copy_image_info(original_img, predicted_catheter_mask)

        # ants.image_write(predicted_catheter_mask, str(config.OUTPUT_DIR / f'subj{subject_id}_taskB.nii.gz'))
        # predicted_catheter_mask = ants.image_read(str(config.OUTPUT_DIR / f'subj{subject_id}_taskB.nii.gz'))

        # catheter_length = run_inference(predicted_catheter_mask.numpy(), predicted_artifact_mask.numpy(), np.array(predicted_catheter_mask.spacing), './models/catheter_hidden_model.pkl',
        #            smoothing_factor=100, downsample_mm=2.0, bridge_dist_mm=2.0)
        
        # output_data.append({
        #     "subject": f'subj{subject_id}',
        #     "length": round(catheter_length, 2)
        # })
    
    # output_df = pd.DataFrame(output_data)
    # output_df.to_csv(f'{config.OUTPUT_DIR}/taskC.csv', index=False)