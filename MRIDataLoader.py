import os
import glob
import json
import nibabel as nib
import matplotlib.pyplot as plt

class MRIDataLoader:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.subjects = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith(".")
        ])

    def _find(self, subject_path, pattern):
        files = glob.glob(os.path.join(subject_path, "**", pattern), recursive=True)
        return files[0] if len(files) > 0 else None

    def _load_meta(self, path):
        if not os.path.exists(path):
            return None

        with open(path, "r") as f:
            meta = json.load(f)

        return meta

    def load_subject(self, idx):
            subject = self.subjects[idx]
            path = os.path.join(self.root_dir, subject)

            mri_path = self._find(path, "*_image.nii.gz")
            art_path = self._find(path, "*_artifact.nii.gz")
            cat_path = self._find(path, "*_catheter.nii.gz")
            meta_path = self._find(path, "*.json")

            mri = nib.load(mri_path).get_fdata()
            artifact = nib.load(art_path).get_fdata() if art_path else None
            catheter = nib.load(cat_path).get_fdata() if cat_path else None
            meta = self._load_meta(meta_path) if meta_path else None

            return {
                "subject": subject,
                "mri": mri,
                "artifact": artifact,
                'catheter_location': meta['hemisphere'] if meta else None,
                "catheter": catheter,
                "catheter_length": meta['catheter_length_mm'] if meta else None
            }

    def show_slice(self, data_dict, z=None, alpha=0.4):
        mri = data_dict["mri"]
        artifact = data_dict["artifact"]
        catheter = data_dict["catheter"]

        if z is None:
            z = mri.shape[2] // 2

        plt.figure(figsize=(6,6))
        plt.imshow(mri[:, :, z], cmap="gray")

        if artifact is not None:
            plt.imshow(artifact[:, :, z], cmap="Reds", alpha=alpha)

        if catheter is not None:
            plt.imshow(catheter[:, :, z], cmap="Blues", alpha=alpha)

        plt.title(f"Subject: {data_dict['subject']} | Slice {z}")
        plt.axis("off")
        plt.show()

    def explore(self, data_dict):
        from ipywidgets import interact

        mri = data_dict["mri"]
        artifact = data_dict["artifact"]
        catheter = data_dict["catheter"]

        def view(z):
            plt.figure(figsize=(6,6))
            plt.imshow(mri[:, :, z], cmap="gray")

            if artifact is not None:
                plt.imshow(artifact[:, :, z], cmap="Reds", alpha=0.4)

            if catheter is not None:
                plt.imshow(catheter[:, :, z], cmap="Blues", alpha=0.4)

            plt.title(f"Slice {z}")
            plt.axis("off")
            plt.show()

        interact(view, z=(0, mri.shape[2]-1))
