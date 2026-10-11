# OCT-RTFL: target annotation pilot

## Purpose and status

Create an expert-reviewed target reference for NFL boundary and thickness evaluation. This pack contains the **20 previously inspected development cases**, not a newly independent test cohort and not 20 confirmed independent patients. No SAM/nnU-Net prediction is supplied as an expert label. This is the first preparation step, not model training.

The eventual reference pilot may be expanded to around 20-30 scans after subject/eye/volume identity and expert availability are known. That is a planning scope, not a guarantee of sufficient data. Independent evaluation must reserve additional unseen groups; do not randomly split these correlated B-scans as if they were independent patients.

## What the expert reviews

- Original-resolution grayscale images in `images/`; palette decoded, no crop/resize/reflection or intensity enhancement added.
- Upper target: ILM. Lower target: NFL/GCL interface. Confirm these definitions with the expert before annotation.
- Trace both interfaces over the evaluable width, including edges and thin regions. A thicker or thinner segment is not automatically an error.
- Mark regions whose interfaces cannot be assessed as ungradable. Distinguish these from regions simply not reviewed yet.
- Preserve the original image and all manual corrections. Record annotator/reviewer, date, tool and comments in `review_log.csv`.

## Annotation tool and exchange format

Use the expert's familiar boundary/polyline annotation tool if one is already available. The supplied CSVs are **blank exchange-format templates**, not a request to type 1,004 numbers manually. Trace boundaries interactively, then export their coordinates; conversion into this schema should be checked against the tool's coordinate convention before use. Tool-specific import/export is the next step once the available annotation tool is identified.

If SAM-assisted annotation is tried later, save the raw SAM result separately from the expert-corrected reference. Log the assistance and human corrections. An unchanged SAM output is not automatically an independent expert reference.

## Coordinates and field definitions

- `column`: original zero-based image column, from 0 to width-1.
- `ilm_y`, `nfl_gcl_y`: boundary coordinates in original image rows; row increases downward, first pixel centre is row 0. Fractional coordinates are allowed. No conversion from original pixels to resized pixels is required.
- Adopt `0 <= ilm_y <= nfl_gcl_y < height` for this exchange schema. If a tool uses pixel-edge coordinates or annotations lie outside the visible image, document that and convert explicitly; do not silently clamp them.
- `status`: `unreviewed`, `gradable` or `ungradable`.
- A `gradable` column requires both finite boundary coordinates and an anatomically reviewed pair.
- An `ungradable` or `unreviewed` column must not be interpreted as background or zero thickness. Leave both coordinate fields empty in the final exchanged column record.
- Coincident expert boundaries may represent zero thickness if the expert explicitly judges that appropriate. Absence of a predicted mask is not evidence of zero thickness.
- Axial thickness is `nfl_gcl_y - ilm_y` in original-image pixels. Physical units require verified axial spacing.

Annotations are geometric interfaces; they are not foreground-pixel indices. If training masks are later rasterized, retain these original curves and document the rasterization separately. Avoid an extra +1 added to manual geometric thickness.

## Gaps, tracing and interpolation

Manual polylines are interpolated only within explicitly annotated, gradable segments when converting to per-column coordinates. Do not connect across an ungradable gap, extrapolate unmarked edges, or smooth interfaces merely to make a prettier profile. Save the original polyline vertices as well as any converted per-column record.

## Group identity before any split

Complete `grouping_to_complete.csv` using verified metadata. Record pseudonymous subject, eye and volume IDs and the evidence supporting the mapping. Keep identifiers consistent across files. Do not manufacture a separate subject for each case ID or infer a volume from adjacent filenames alone. Unknown entries remain blank. This pack creates no train/validation/test assignment.

## Completion check before training/evaluation

1. Confirm image identity, original dimensions and orientation.
2. Complete provenance and reviewer fields; an export filename is not proof of expert review.
3. Overlay exported coordinates on the unchanged image, inspecting full width and edges.
4. Check ordered finite boundary pairs on every gradable column; report ungradable/unreviewed counts.
5. Reconcile the annotation tool's coordinates with the exchange schema.
6. Establish group-disjoint development/evaluation allocation on an expanded cohort.
7. Only then compare or fine-tune nnU-Net and a DINO encoder/decoder; disclose whether SAM results require user interaction.

Lower-boundary MAE, thickness MAE/bias and coverage will be key measurements. High mask extractability, high model scores and smooth-looking curves alone are not accuracy evidence.
