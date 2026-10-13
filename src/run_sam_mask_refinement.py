import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


root = Path.home() / "OCT-RTFL"
job_id = os.environ.get("PBS_JOBID")
if not job_id:
    raise RuntimeError("Submit through PBS; do not run on the login node.")

# Use the GPU exposed by the scheduler; never select another GPU.
if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
    raise RuntimeError("No CUDA_VISIBLE_DEVICES allocation found. Share this log.")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise RuntimeError("Expected exactly one scheduler-visible CUDA GPU.")

checkpoint = Path.home() / "research_tools/checkpoints/sam2.1_hiera_small.pt"
expected_hash = "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
if sha256_file(checkpoint) != expected_hash:
    raise RuntimeError("Checkpoint differs from the previously downloaded file.")

prompt_path = root / "outputs/sam_prompt_pilot/runs/27208.rachel/prompts.json"
prompt_record = json.loads(prompt_path.read_text())
cases = prompt_record["cases"]
expected_cases = {
    "target_00512", "target_00397", "target_00569", "target_00170"
}
if len(cases) != 4 or {c["case_id"] for c in cases} != expected_cases:
    raise RuntimeError("Unexpected pilot case list.")

out = root / "outputs/sam_mask_refinement/runs" / job_id
out.mkdir(parents=True, exist_ok=False)
shutil.copy2(prompt_path, out / "prompts.json")
shutil.copy2(__file__, out / "inference_script.py")

repo = Path.home() / "research_tools/sam2"
revision = subprocess.check_output(
    ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
).strip()

config = "configs/sam2.1/sam2.1_hiera_s.yaml"
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
run_record = {
    "status": "running",
    "job_id": job_id,
    "python": sys.executable,
    "torch": torch.__version__,
    "sam_revision": revision,
    "checkpoint_sha256": expected_hash,
    "config": config,
    "gpu": torch.cuda.get_device_name(0),
    "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
    "autocast_dtype": str(dtype),
    "postprocessing": False,
    "method": "selected_mask_plus_same_points_single_pass",
    "parent_run": "27208.rachel",
    "selected_parent_candidate": 1,
    "selection_basis": "manual visual development review",
    "additional_clicks": 0,
    "ground_truth_available": False,
}
(out / "run_record.json").write_text(json.dumps(run_record, indent=2))

free, total = torch.cuda.mem_get_info()
print("GPU:", run_record["gpu"], flush=True)
print(f"Free VRAM: {free / 2**30:.2f} / {total / 2**30:.2f} GiB", flush=True)

model = build_sam2(
    config, str(checkpoint),
    device="cuda", apply_postprocessing=False,
)
predictor = SAM2ImagePredictor(
    model, max_hole_area=0, max_sprinkle_area=0,
)
results = []

for case in cases:
    case_id = case["case_id"]
    path = Path(case["image_path"])
    if sha256_file(path) != case["image_sha256"]:
        raise RuntimeError(f"Input image changed: {case_id}")

    with Image.open(path) as image:
        if image.mode != "L" or image.size != (1004, 800):
            raise RuntimeError(f"Unexpected input format: {case_id}")
        gray = np.array(image)
    rgb = np.repeat(gray[..., None], 3, axis=2)

    points = np.asarray(case["points"], dtype=np.float32)
    labels = np.asarray(case["labels"], dtype=np.int32)
    if points.shape != (8, 2) or labels.tolist() != [1]*4 + [0]*4:
        raise RuntimeError(f"Unexpected prompts: {case_id}")
    if not (
        np.isfinite(points).all()
        and ((points[:, 0] >= 0) & (points[:, 0] < 1004)).all()
        and ((points[:, 1] >= 0) & (points[:, 1] < 800)).all()
    ):
        raise RuntimeError(f"Invalid coordinates: {case_id}")

    print("Predicting:", case_id, flush=True)
    parent_file = (
        root / "outputs/sam_prompt_pilot/runs/27208.rachel"
        / f"{case_id}_candidates.npz"
    )
    with np.load(parent_file, allow_pickle=False) as parent:
        prior_logits = parent["low_res_logits"][0:1].astype(np.float32)

    if prior_logits.shape != (1, 256, 256):
        raise RuntimeError(f"Unexpected prior shape: {case_id}")
    if not np.isfinite(prior_logits).all():
        raise RuntimeError(f"Non-finite prior: {case_id}")

    with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
        predictor.set_image(rgb)
        masks, scores, low_res_logits = predictor.predict(
            point_coords=points,
            point_labels=labels,
            mask_input=prior_logits,
            multimask_output=False,
        )

    if masks.shape != (1, 800, 1004) or not np.isfinite(scores).all():
        raise RuntimeError(f"Unexpected SAM output: {case_id}")
    masks = masks.astype(bool)

    np.savez_compressed(
        out / f"{case_id}_candidates.npz",
        masks=masks,
        scores=scores,
        low_res_logits=low_res_logits,
    )

    # Three alternatives: full scan above, upper-region zoom below.
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 10), constrained_layout=True, squeeze=False
    )
    for k, mask in enumerate(masks):
        Image.fromarray(mask.astype(np.uint8) * 255).save(
            out / f"{case_id}_candidate_{k+1}.png"
        )
        for row in range(2):
            ax = axes[row, k]
            ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
            overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
            overlay[mask] = [1.0, 0.55, 0.0, 0.25]
            ax.imshow(overlay)
            if mask.any() and not mask.all():
                ax.contour(mask.astype(float), levels=[0.5],
                           colors=["orange"], linewidths=0.6)
            for label, color, marker in [(1, "lime", "+"), (0, "red", "x")]:
                selected = points[labels == label]
                ax.scatter(selected[:, 0], selected[:, 1],
                           c=color, marker=marker, s=28)
            ax.set_xlim(0, 1003)
            ax.set_ylim(799, 0) if row == 0 else ax.set_ylim(480, 100)
            ax.set_title(
                f"Candidate {k+1} | model score {float(scores[k]):.3f}"
                + ("\nFull scan" if row == 0 else "\nUpper-region zoom")
            )
            ax.set_xlabel("Original column")
            ax.set_ylabel("Original row")

        results.append({
            "case_id": case_id,
            "candidate": k + 1,
            "model_score": float(scores[k]),
            "foreground_pixels": int(mask.sum()),
        })

    fig.suptitle(f"{case_id} — SAM mask-guided refinement; scores are not accuracy")
    fig.savefig(out / f"{case_id}_review.png", dpi=160)
    plt.close(fig)
    predictor.reset_predictor()

(out / "candidate_summary.json").write_text(json.dumps(results, indent=2))
run_record["status"] = "complete"
run_record["completed_cases"] = len(cases)
(out / "run_record.json").write_text(json.dumps(run_record, indent=2))
print("SUCCESS: four scans; one refinement mask per scan.", flush=True)
print("Results:", out, flush=True)
