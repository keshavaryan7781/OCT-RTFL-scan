from pathlib import Path
import json

import numpy as np
import pandas as pd
from PIL import Image
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_NAME = "Dataset501_DukeNFL"


def build_nfl_mask(boundaries, image_shape):
    """Convert expert boundary coordinates into 0/1/2 training labels."""
    height, width = image_shape
    boundaries = np.asarray(boundaries, dtype=np.float64)

    if boundaries.shape != (8, width):
        raise ValueError(f"Unexpected boundary shape: {boundaries.shape}")

    pair = boundaries[[0, 1], :]

    present = (
        np.isfinite(pair).all(axis=0)
        & (pair >= 1).all(axis=0)
        & (pair <= height).all(axis=0)
    )

    top, bottom = pair - 1.0
    crossed = present & (bottom < top)
    valid_columns = present & ~crossed

    if crossed.any():
        raise ValueError("Crossed NFL boundaries found; review this case.")

    if not valid_columns.any():
        raise ValueError("No valid NFL annotation columns.")

    rows = np.arange(height)[:, None]

    inside = (
        valid_columns[None, :]
        & (rows >= top[None, :])
        & (rows < bottom[None, :])
    )

    labels = np.full((height, width), 2, dtype=np.uint8)
    labels[:, valid_columns] = 0
    labels[inside] = 1

    return labels, valid_columns


def as_uint8(image):
    """Allow lossless conversion only; never silently rescale intensities."""
    if image.dtype == np.uint8:
        return image

    if (
        not np.isfinite(image).all()
        or image.min() < 0
        or image.max() > 255
        or not np.equal(image, np.rint(image)).all()
    ):
        raise ValueError("Image cannot be converted losslessly to uint8.")

    return image.astype(np.uint8)


def save_json(path, content):
    path.write_text(json.dumps(content, indent=2) + "\n")

def main():
    split = json.loads(
        (PROJECT_ROOT / "configs" / "subject_split.json").read_text()
    )

    manifest = pd.read_csv(
        PROJECT_ROOT / "outputs" / "audit" / "duke_manifest.csv"
    )

    partitions = ("train", "val", "test")
    subject_to_split = {}

    for partition in partitions:
        for subject_id in split[partition]:
            if subject_id in subject_to_split:
                raise ValueError(f"Subject appears twice: {subject_id}")
            subject_to_split[subject_id] = partition

    if set(manifest["subject_id"]) != set(subject_to_split):
        raise ValueError("Manifest subjects do not match the saved split.")

    if not manifest["case_id"].is_unique:
        raise ValueError("Duplicate case identifiers.")

    # Derive membership from the frozen subject split.
    manifest["split"] = manifest["subject_id"].map(subject_to_split)

    counts = manifest.groupby("split").size().to_dict()
    expected = {"train": 66, "val": 22, "test": 22}

    if counts != expected:
        raise ValueError(f"Unexpected case counts: {counts}")

    cases = {
        partition: sorted(
            manifest.loc[
                manifest["split"] == partition, "case_id"
            ].tolist()
        )
        for partition in partitions
    }

    dataset_root = PROJECT_ROOT / "nnUNet_raw" / DATASET_NAME
    preprocessed_root = (
        PROJECT_ROOT / "nnUNet_preprocessed" / DATASET_NAME
    )

    # Do not mix this export with an older or partial export.
    if dataset_root.exists():
        raise FileExistsError(
            f"{dataset_root} already exists. Review before re-exporting."
        )

    fold_split = [{"train": cases["train"], "val": cases["val"]}]
    split_file = preprocessed_root / "splits_final.json"

    if split_file.exists():
        if json.loads(split_file.read_text()) != fold_split:
            raise ValueError("Existing nnU-Net split differs from our split.")

    for folder in ("imagesTr", "labelsTr", "imagesTs", "labelsTs"):
        (dataset_root / folder).mkdir(parents=True, exist_ok=False)

    duke_root = PROJECT_ROOT / "duke_dataset" / "2015_BOE_Chiu"

    for subject_id, rows in manifest.groupby("subject_id", sort=True):
        data = loadmat(
            duke_root / f"{subject_id}.mat",
            variable_names=["images", "manualLayers1"],
            squeeze_me=False,
        )

        volume = data["images"]
        boundaries = data["manualLayers1"]

        if volume.ndim != 3:
            raise ValueError(f"{subject_id}: invalid volume shape")

        height, width, n_slices = volume.shape

        if boundaries.shape != (8, width, n_slices):
            raise ValueError(f"{subject_id}: annotation shape mismatch")

        for row in rows.itertuples(index=False):
            scan_idx = int(row.slice_index)
            case_id = row.case_id

            if not 0 <= scan_idx < n_slices:
                raise ValueError(f"{case_id}: invalid slice index")

            image = as_uint8(volume[:, :, scan_idx])

            labels, valid_columns = build_nfl_mask(
                boundaries[:, :, scan_idx],
                image.shape,
            )

            # Check that export reproduces our notebook audit.
            if int(valid_columns.sum()) != int(row.valid_columns):
                raise ValueError(f"{case_id}: valid-column count changed")

            if int((labels == 1).sum()) != int(row.nfl_pixels):
                raise ValueError(f"{case_id}: NFL pixel count changed")

            suffix = "Ts" if row.split == "test" else "Tr"

            image_path = (
                dataset_root / f"images{suffix}" / f"{case_id}_0000.png"
            )
            label_path = (
                dataset_root / f"labels{suffix}" / f"{case_id}.png"
            )

            Image.fromarray(image).save(image_path)
            Image.fromarray(labels).save(label_path)

            # Confirm label values survive PNG serialization.
            with Image.open(label_path) as saved:
                if not np.array_equal(np.asarray(saved), labels):
                    raise RuntimeError(f"{case_id}: saved label mismatch")

        print(f"{subject_id}: exported {len(rows)} scans", flush=True)
        del data, volume, boundaries

    dataset_json = {
        "channel_names": {"0": "OCT"},
        "labels": {
            "background": 0,
            "NFL": 1,
            "ignore": 2,
        },
        "numTraining": len(cases["train"]) + len(cases["val"]),
        "file_ending": ".png",
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }

    save_json(dataset_root / "dataset.json", dataset_json)

    preprocessed_root.mkdir(parents=True, exist_ok=True)
    save_json(split_file, fold_split)

    for suffix, expected_count in (("Tr", 88), ("Ts", 22)):
        n_images = len(list((dataset_root / f"images{suffix}").glob("*.png")))
        n_labels = len(list((dataset_root / f"labels{suffix}").glob("*.png")))

        if n_images != expected_count or n_labels != expected_count:
            raise RuntimeError(f"Export count mismatch for {suffix}")

    print("\nExport complete:", counts)
    print("Dataset:", dataset_root)
    print("Custom fold 0 split:", split_file)


if __name__ == "__main__":
    main()