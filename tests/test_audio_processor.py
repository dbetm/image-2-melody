import pytest

from image_to_melody.audio_processor import group_notes


@pytest.mark.parametrize(
    "notes, expected",
    [
        ([57, 61, 64, 55, 59, 62], [55, 57, 59, 61, 62, 64]),
        ([2, 2, 3, 1, 1], [1, 2, 3])
    ]
)
def test_group_notes(notes: list, expected: list) -> list:
    groupped_notes = group_notes(notes)

    assert groupped_notes == expected