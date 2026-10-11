# SAM pilot environment

Environment: oct_sam
Purpose: prompted SAM feasibility experiments for OCT NFL segmentation.
Existing nnU-Net work uses the separate oct_nfl environment.

SAM source repository: https://github.com/facebookresearch/sam2
Local source location: ~/research_tools/sam2
Exact source revision: sam2_commit.txt
Installed package snapshot: requirements.freeze.txt

Installation used SAM2_BUILD_CUDA=0 to skip the optional CUDA extension.
SAM imports passed. GPU inference validation is still pending.

Planned checkpoint: sam2.1_hiera_small.pt
Checkpoint directory: ~/research_tools/checkpoints

The requirements snapshot may contain an editable SAM installation path.
Recreating this environment requires cloning SAM at the recorded revision
and installing it from the appropriate local path.

This setup record does not establish target segmentation accuracy.
