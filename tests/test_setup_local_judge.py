import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from scripts.setup_local_judge import download_checked


def test_checked_download_preserves_existing_assets_and_rejects_wrong_bytes():
    with TemporaryDirectory() as folder:
        root = Path(folder)
        source = root / "source.bin"
        source.write_bytes(b"trusted model bytes")
        checksum = hashlib.sha256(source.read_bytes()).hexdigest()
        target = root / "model.bin"
        download_checked(source.as_uri(), target, checksum)
        assert target.read_bytes() == source.read_bytes()
        download_checked("https://unused.invalid/", target, checksum)
        with pytest.raises(ValueError, match="existing"):
            download_checked(source.as_uri(), target, "0" * 64)
        assert target.read_bytes() == source.read_bytes()
        bad = root / "bad.bin"
        with pytest.raises(ValueError, match="Downloaded"):
            download_checked(source.as_uri(), bad, "0" * 64)
        assert not bad.exists()
