# OCT-RTFL - annotation pilot preparation

This toolkit prepares the next step after the 8 October 2026 target full-width review. It does not install SAM/DINO, launch GPU work, infer labels, or connect to the HPC.

## Copy and run on the HPC

Copy this entire `OCT-RTFL-annotation-pilot` directory to:

```text
~/OCT-RTFL/tools/OCT-RTFL-annotation-pilot/
```

Open `notebooks/04_prepare_target_annotation.ipynb` inside that directory in your existing `oct_nfl` kernel. Run cells in order. The notebook reads the original native pilot manifest and scans, so moving only the notebook will not work. It uses Pillow and Python standard libraries; the notebook preview uses pandas/matplotlib already used in the prior inspection notebooks.

Equivalent terminal commands, after activating the existing environment:

```bash
cd ~/OCT-RTFL
python tools/OCT-RTFL-annotation-pilot/src/prepare_target_annotation.py audit
python tools/OCT-RTFL-annotation-pilot/src/prepare_target_annotation.py prepare
```

The audit reads the 20 original source BMPs, verifies their saved hashes, decodes the palette and checks exact agreement with the previous native grayscale PNGs. It inventories filenames in `INT`, if present; it does not assume these are labels or patient metadata. It counts the BMP files without performing a full 1,025-image decode.

## Output

```text
outputs/target_annotation/pilot_v1/
  images/                         # Original-size decoded grayscale PNGs
  boundary_templates/             # Blank per-column CSVs, NOT ground truth
  case_manifest.csv               # Identity, image dimensions and hashes
  grouping_to_complete.csv        # Verified subject/eye/volume mapping needed
  review_log.csv                  # Annotator/reviewer/date/tool/status
  audit.json
  PACK_STATUS.json
  ANNOTATION_PROTOCOL.md
```

No model overlay is included in the reference images. Existing predictions, checkpoints and inputs are never modified. A pre-existing destination causes preparation to stop without overwriting it. Keep the existing pack if already created; use a new named output only for a deliberately separate pack. A failure during writing cleans the temporary directory, not existing project files.

## Current scope

The existing 20 cases are explicitly development-only. They are useful for initial expert discussion and annotation workflow setup, but are not a fresh independent test set. Additional examples and group-disjoint allocation depend on real metadata. No target accuracy percentage can be reported from these blank templates.

Next required information: an available expert/annotation tool or existing target annotations, plus available subject/eye/volume metadata. The notebook makes this information easier to inspect; it does not invent missing labels or grouping.
