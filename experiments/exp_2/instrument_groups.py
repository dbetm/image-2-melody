from image_to_melody.audio_processor import MidiChords as Chords, group_notes


class Instrument:
    def __init__(
        self,
        name: str,
        midi_num: int,
        notes: list,
        color_channel: str = "avg_all",
        volume: int = 64,
    ):
        self.name = name

        assert midi_num >= 0 and midi_num <= 127, "Select a valid MIDI number"
        self.midi_num = midi_num

        self.notes = notes
        assert (
            volume >= 0 and volume <= 127,
            "Select a valid volume (0-127). 64 is recommended"
        )
        self.volume = volume
        assert color_channel in {"red", "green", "blue", "avg_all"}
        self.color_channel = color_channel


RED_GROUP = {
    "description": (
        "Electric guitar and Synth pad, which are mapped to red tones and thus to"
        " energy and excitement."
    ),
    "instruments": [
        Instrument(
            "electric guitar", 27, group_notes(Chords.G_MAJOR + Chords.C_MAJOR), "red", 100
        ),
        Instrument("pad 6 (metallic)", 93, Chords.F_MAJOR)
    ]
}


YELLOW_GROUP = {
    "description": (
        "Piano, acustic guitar, and bass, which are mapped to yellow tones and thus to"
        " happiness and joy."
    ),
    "instruments": [
        Instrument("piano", 0, Chords.C_MAJOR, "green", 100),
        Instrument("acustic guitar", 24, Chords.G_MAJOR, "red", 74),
        Instrument(
            name="bass",
            midi_num=32,
            notes=group_notes(Chords.C_MAJOR_ROOT + Chords.F_MAJOR_ROOT + Chords.G_MAJOR_ROOT),
            color_channel="blue",
        )
    ]
}


GREEN_GROUP = {
    "description": (
        "Viola, trombone, and banjo, which are mapped to green tones and thus to"
        " growth and renewal."
    ),
    "instruments": [
        Instrument("viola", 41, group_notes(Chords.B_MAJOR + Chords.C_MAJOR), "green", 100),
        Instrument("trombone", 57, Chords.C_MAJOR, "red"),
        Instrument(
            name="banjo",
            midi_num=105,
            notes=Chords.E_MAJOR,
            color_channel="blue",
        )
    ]
}


BLUE_2_GROUP = {
    "description": (
        "Drum, bass, whistle, and brass, which are mapped to clear blue tones and thus to"
        " energy and vitality."
    ),
    "instruments": [
        Instrument("drum", 117, Chords.A_MINOR, volume=84, color_channel="blue",),
        Instrument("whistle", 78, Chords.A_MINOR, "green"),
        Instrument("brass", 61, Chords.C_MAJOR, "red"),
    ]
}


BLUE_GROUP = {
    "description": (
        "Violin, clarinet, and piano, which are mapped to blue tones and thus to"
        " sadness and melancholy."
    ),
    "instruments": [
        Instrument("violin", 40, Chords.B_MINOR, "blue", 100),
        Instrument("clarinet", 71, group_notes(Chords.A_MINOR + Chords.F_MINOR), "green", 60),
        Instrument("piano", 3, Chords.C_MAJOR, "red"),
    ]
}


PURPLE_GROUP = {
    "description": (
        "Synth FX 4 (atmosphere), ethnic Koto, and guitar electric, which are mapped"
        " to purple tones and thus to creativity and imagination."
    ),
    "instruments": [
        Instrument("synth FX 4", 99, Chords.E_MINOR, "blue", 100),
        Instrument("ethnic koto", 107, group_notes(Chords.B_MINOR + Chords.D_MINOR), "green"),
        Instrument("electric guitar", 26, Chords.E_MAJOR, "red"),
    ]
}


PINK_GROUP = {
    "description": (
        "Harp, flute, and choir, which are mapped to pink tones and thus to softness."
    ),
    "instruments": [
        Instrument("harp", 46, Chords.G_MAJOR, "red", 100),
        Instrument("flute", 73, group_notes(Chords.C_MAJOR + Chords.G_MAJOR), "green"),
        Instrument("choir", 52, Chords.G_MAJOR, "blue"),
    ]
}


RED_2_GROUP = {
    "description": (
        "Trumpet, alto sax and drum, which are mapped to red tones and thus to anger"
        " and aggression."
    ),
    "instruments": [
        Instrument(
            "trumpet", 56, group_notes(Chords.C_MAJOR + Chords.E_MAJOR), "red", 84
        ),
        Instrument("alto sax", 65, group_notes(Chords.A_MAJOR + Chords.F_MAJOR), "blue", 64),
        Instrument("drum", 117, group_notes(Chords.E_MAJOR + Chords.F_MAJOR), volume=84)
    ]
}

