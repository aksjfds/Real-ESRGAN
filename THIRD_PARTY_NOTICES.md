# Third-party notice

`inference/basicvsrpp.py` is an inference-oriented, dependency-light adaptation of OpenMMLab MMagic's `BasicVSRPlusPlusNet` and supporting BasicVSR/SPyNet components. MMagic is distributed under the Apache License 2.0.

Upstream project: https://github.com/open-mmlab/mmagic

`inference/rife425.py` contains an inference-only adaptation of the Practical-RIFE 4.25 IFNet/warp architecture. Practical-RIFE and its released trained-model content are distributed under the MIT License.

Upstream project: https://github.com/hzwer/Practical-RIFE

The RIFE 4.25 `flownet.pkl` is downloaded from the upstream project's official Google Drive release on first use and cached under `~/.cache/realesrgan/rife-v4.25/`. It is not stored in this repository.

The APISR branch downloads the official APISR source at the pinned commit recorded in `inference/apisr_backend.py` and imports its GRL architecture at runtime. The APISR source and released model weights are not stored in this repository. APISR is published by its upstream authors under the GPL-3.0 license and its published disclaimer/weight terms remain applicable to those downloaded materials.

Upstream project: https://github.com/Kiteretsu77/APISR

v8.1 audio separation uses the `audio-separator` Python package, distributed under the MIT License, as an isolated runtime wrapper for UVR-compatible source-separation models.

Upstream project: https://github.com/nomadkaraoke/python-audio-separator

The selected `vocals_mel_band_roformer.ckpt` model and its companion metadata are downloaded by `audio-separator` on first preparation into the external Real-ESRGAN audio cache. They are not stored in this repository. Model provenance and any model-specific terms remain those published by the upstream audio-separator / UVR model catalog.
