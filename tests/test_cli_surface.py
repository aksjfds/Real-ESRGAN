from pathlib import Path


def test_unimplemented_backends_and_presets_are_absent_from_cli():
    source = (Path(__file__).parents[1] / "realesrgan.py").read_text()
    for option in (
        '"--preprocess"',
        '"--preprocess-model-path"',
        '"--cugan-ensemble"',
        '"--cugan-model-path"',
        '"compressed-anime"',
        '"blurred-anime"',
        '"max"',
    ):
        assert option not in source
