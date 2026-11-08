import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "anime_dub_tools"
FORBIDDEN = ("Wangan", "Midnight", "Akio", "Asakura")


def test_package_has_no_show_roster():
    text = "\n".join(path.read_text(encoding="utf-8") for path in PACKAGE.rglob("*.py"))
    for word in FORBIDDEN:
        assert word not in text
    assert "VoiceStudio" not in text
    assert "debpalash" not in text


@pytest.mark.live_models
@pytest.mark.skipif(
    os.environ.get("ANIME_DUB_LIVE") != "1",
    reason="set ANIME_DUB_LIVE=1 to import optional model packages; weights are not downloaded here",
)
def test_live_backends_import():
    import audio_separator.separator  # noqa: F401
    import pyannote.audio  # noqa: F401
    import indextts.infer_v2_5  # noqa: F401
