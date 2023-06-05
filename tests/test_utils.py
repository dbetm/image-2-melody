import pytest

from image_to_melody.utils import map_value_to_dest



@pytest.mark.parametrize(
    "val, dest_vals, max_value, expected",
    [
        (25, [56, 58, 64, 66], 255, 56),
        (0, [220.00, 246.94, 261.63, 293.66, 329.63, 366.67, 440.00], 180, 220.00),
        (255, [57, 61, 64], 255, 64),
    ]
)
def test_map_value_to_dest_using_max_value(val, dest_vals, max_value, expected):
    result = map_value_to_dest(val_=val, dest_vals=dest_vals, max_value=max_value)

    assert result == expected


@pytest.mark.parametrize(
    "val, dest_vals, thresholds, expected",
    [
        (25, [56, 58, 64, 66], [63.75, 127.5, 191.25, 255], 56),
        (
            0,
            [220.00, 246.94, 261.63, 293.66, 329.63, 366.67, 440.00], 
            [25.71, 51.42, 77.14, 102.85, 128.57, 154.28, 180.0],
            220.00
        ),
        (255, [57, 61, 64], [85.0, 170.0, 255.0], 64),
    ]
)
def test_map_value_to_dest_using_thresholds(val, dest_vals, thresholds, expected):
    result = map_value_to_dest(val_=val, dest_vals=dest_vals, thresholds=thresholds)

    assert result == expected