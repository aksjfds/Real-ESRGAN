from dataclasses import fields

from enhance.pipeline import PipelineConfig


def test_unimplemented_model_backends_are_not_exposed():
    names = {field.name for field in fields(PipelineConfig)}
    assert not any("preprocess" in name for name in names)
    assert not any("cugan" in name for name in names)
