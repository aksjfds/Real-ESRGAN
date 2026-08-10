# Third-party notice

`inference/basicvsrpp.py` is an inference-oriented, dependency-light adaptation of OpenMMLab MMagic's `BasicVSRPlusPlusNet` and supporting BasicVSR/SPyNet components. MMagic is distributed under the Apache License 2.0.

Upstream project: https://github.com/open-mmlab/mmagic

`inference/rife425.py` contains an inference-only adaptation of the Practical-RIFE 4.25 IFNet/warp architecture. Practical-RIFE and its released trained-model content are distributed under the MIT License.

Upstream project: https://github.com/hzwer/Practical-RIFE

The RIFE 4.25 `flownet.pkl` is downloaded from the upstream project's official Google Drive release on first use and cached under `~/.cache/realesrgan/rife-v4.25/`. It is not stored in this repository.
