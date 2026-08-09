# Inference weights

Real-ESRGAN weights in this directory are regular repository assets. The BasicVSR++ NTIRE 2021 compressed-video Track 1 checkpoint is intentionally tracked with Git LFS at:

`inference/weights/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth`

Official source:

`https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c128n25_ntire_decompress_track1_20210223-7b2eba02.pth`

Run `tools/vendor_basicvsrpp.sh` from a machine with Git LFS and network access to download and stage the checkpoint. Until the LFS object has been committed and pushed, the Python runtime keeps its existing official-download fallback.
