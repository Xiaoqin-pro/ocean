import numpy as np
import pytest

from reliability.quality_descriptors import DESCRIPTOR_NAMES, image_quality_descriptors
from reliability.quality_grouping import fit_quality_grouping


def test_quality_descriptors_are_input_only_and_finite():
    image = np.full((16, 16, 3), (20, 80, 140), dtype=np.uint8)
    values = image_quality_descriptors(image)
    assert values.shape == (len(DESCRIPTOR_NAMES),)
    assert np.isfinite(values).all()


def test_quality_grouping_is_deterministic_and_validates_shape():
    descriptors = np.vstack([np.zeros((4, 8)), np.ones((4, 8)), np.full((4, 8), 3.0)])
    grouping = fit_quality_grouping(descriptors, groups=3, seed=20260722)
    assert np.array_equal(grouping.predict(descriptors), grouping.predict(descriptors.copy()))
    with pytest.raises(ValueError):
        grouping.predict(np.zeros((3, 7)))

