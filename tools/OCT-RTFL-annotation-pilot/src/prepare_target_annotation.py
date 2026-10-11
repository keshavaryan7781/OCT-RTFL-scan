"""Prepare original-resolution target images for expert annotation.

No GPU, model inference, pseudo-labels or automatic patient grouping.
Run from ~/OCT-RTFL, or supply --project-root explicitly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageChops


PILOT_IDS = [f"target_{s}" for s in (
    "00512", "00000", "00056", "00113", "00170", "00227", "00284",
    "00341", "00397", "00454", "00511", "00569", "00626", "00683",
    "00739", "00796", "00853", "00910", "00967", "01024",
)]
CASE_PATTERN = re.compile(r"target_[0-9]{5}\Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_path(raw: str, root: Path) -> Path:
    path = Path(raw).expanduser()
    return path if path.is_absolute() else root / path


def decode_gray(path: Path) -> Image.Image:
    """Decode the palette; do not reinterpret palette indices as intensity."""
    with Image.open(path) as source:
        rgba = source.convert("RGBA")
        rgba.load()
    red, green, blue, alpha = rgba.split()
    if alpha.getextrema() != (255, 255):
        raise ValueError(f"Non-opaque pixels: {path}")
    if ImageChops.difference(red, green).getbbox() or ImageChops.difference(red, blue).getbbox():
        raise ValueError(f"Non-grayscale used colors: {path}")
    if red.getextrema()[0] == red.getextrema()[1]:
        raise ValueError(f"Constant decoded image: {path}")
    return red


def read_manifest(root: Path) -> tuple[Path, list[dict]]:
    manifest = root / "outputs/target_pilot/pilot_export_manifest.csv"
    if not manifest.is_file():
        raise FileNotFoundError(f"Existing native pilot manifest required: {manifest}")
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        required = {"case_id", "source_path", "input_path", "source_sha256", "width", "height"}
        if required - fields:
            raise ValueError(f"Manifest schema differs; missing {sorted(required - fields)}. Do not guess replacements.")
        rows = list(reader)
    ids = [r["case_id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs in the pilot manifest.")
    if set(ids) != set(PILOT_IDS):
        raise ValueError("Manifest case set differs from the documented 20-case development pilot.")
    for row in rows:
        if not CASE_PATTERN.fullmatch(row["case_id"]):
            raise ValueError("Unsafe/unknown case ID.")
        for name in ("source_path", "input_path"):
            if not row[name].strip() or not resolve_path(row[name], root).is_file():
                raise FileNotFoundError(f"{row['case_id']}: unavailable {name}: {row[name]}")
        if not re.fullmatch(r"[a-fA-F0-9]{64}", row["source_sha256"]):
            raise ValueError(f"{row['case_id']}: invalid or missing source SHA-256.")
    return manifest, rows


def audit_project(project_root: str | Path) -> dict:
    root = Path(project_root).expanduser().resolve()
    manifest, rows = read_manifest(root)
    cases, profiles = [], Counter()
    for row in rows:
        source = resolve_path(row["source_path"], root)
        native = resolve_path(row["input_path"], root)
        actual_hash = sha256(source)
        if actual_hash.lower() != row["source_sha256"].lower():
            raise ValueError(f"Source hash mismatch: {row['case_id']}")
        gray = decode_gray(source)
        expected_size = (int(row["width"]), int(row["height"]))
        if gray.size != expected_size:
            raise ValueError(f"Source/native manifest dimensions disagree: {row['case_id']}")
        with Image.open(native) as previous:
            previous.load()
            if previous.mode != "L" or previous.size != gray.size:
                raise ValueError(f"Expected native grayscale input: {native}")
            if ImageChops.difference(gray, previous).getbbox():
                raise ValueError(f"Native input differs from palette-decoded source: {row['case_id']}")
        profiles[f"{gray.height}x{gray.width}"] += 1
        cases.append({
            "case_id": row["case_id"], "source_path": str(source),
            "native_input_path": str(native), "width": gray.width, "height": gray.height,
            "source_sha256": actual_hash,
            "source_parent": str(source.parent.relative_to(root)) if source.is_relative_to(root) else str(source.parent),
        })
    target = root / "raw_data/retina_dataset"
    # Inventory only. Directory names are not automatically treated as patients.
    int_dir = target / "INT"
    int_files = sorted(p for p in int_dir.rglob("*") if p.is_file()) if int_dir.is_dir() else []
    bmp_root = target / "BMP"
    bmp_count = sum(p.is_file() and p.suffix.lower() == ".bmp" for p in bmp_root.rglob("*")) if bmp_root.is_dir() else None
    return {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root), "source_manifest": str(manifest),
        "source_manifest_sha256": sha256(manifest), "verified_cases": len(cases),
        "native_profiles_height_x_width": dict(profiles), "bmp_file_count": bmp_count,
        "metadata_inventory": {
            "INT_present": int_dir.is_dir(), "INT_file_count": len(int_files),
            "INT_suffix_counts": dict(Counter(p.suffix.lower() or "<no suffix>" for p in int_files)),
            "INT_examples_relative_to_target": [str(p.relative_to(target)) for p in int_files[:12]],
            "note": "Filenames only; file contents and patient grouping have not been interpreted.",
        },
        "grouping_status": "unknown; requires verified subject/eye/volume mapping",
        "experiment_role": "development_annotation_pilot; not independent test data",
        "cases": cases,
    }


def write_csv(path: Path, fields: list[str], rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_pack(project_root: str | Path, output: str | Path | None = None) -> dict:
    root = Path(project_root).expanduser().resolve()
    dest = resolve_path(str(output), root) if output else root / "outputs/target_annotation/pilot_v1"
    if dest.exists():
        raise FileExistsError(f"Pack already exists; preserved unchanged: {dest}")
    audit = audit_project(root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".annotation-pilot-", dir=dest.parent))
    try:
        (stage / "images").mkdir()
        (stage / "boundary_templates").mkdir()
        records = []
        for case in audit["cases"]:
            cid = case["case_id"]
            image_path = stage / "images" / f"{cid}.png"
            decode_gray(Path(case["source_path"])).save(image_path)
            record = {**case, "image_file": f"images/{cid}.png", "image_sha256": sha256(image_path)}
            records.append(record)
            write_csv(stage / "boundary_templates" / f"{cid}.csv",
                      ["case_id", "column", "ilm_y", "nfl_gcl_y", "status"],
                      ({"case_id": cid, "column": x, "ilm_y": "", "nfl_gcl_y": "", "status": "unreviewed"}
                       for x in range(case["width"])))
        write_csv(stage / "case_manifest.csv", list(records[0]), records)
        write_csv(stage / "grouping_to_complete.csv",
                  ["case_id", "subject_id", "eye_id", "volume_id", "grouping_evidence", "role"],
                  ({"case_id": c["case_id"], "role": "development"} for c in records))
        write_csv(stage / "review_log.csv",
                  ["case_id", "annotator", "reviewer", "review_date", "review_status", "annotation_tool", "notes"],
                  ({"case_id": c["case_id"], "review_status": "not_started"} for c in records))
        (stage / "audit.json").write_text(json.dumps(audit, indent=2))
        protocol = Path(__file__).resolve().parents[1] / "ANNOTATION_PROTOCOL.md"
        if not protocol.is_file():
            raise FileNotFoundError(f"Keep the supplied toolkit directory intact: missing {protocol}")
        shutil.copy2(protocol, stage / "ANNOTATION_PROTOCOL.md")
        (stage / "PACK_STATUS.json").write_text(json.dumps({
            "status": "awaiting_expert_annotation", "ground_truth_available": False,
            "model_predictions_included": False, "images": len(records),
            "coordinate_system": "original image; zero-based column and row; row increases downward",
            "boundary_templates": "Blank exchange-format templates; not training masks or expert labels.",
            "split_status": "No train/validation/test split created. Existing 20 cases are development-only.",
        }, indent=2))
        if dest.exists():
            raise FileExistsError(f"Output appeared during preparation; refusing overwrite: {dest}")
        stage.rename(dest)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return {"output": str(dest), "images": len(records), "status": "awaiting_expert_annotation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "prepare"])
    parser.add_argument("--project-root", type=Path, default=Path.home() / "OCT-RTFL")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_project(args.project_root) if args.action == "audit" else prepare_pack(args.project_root, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
