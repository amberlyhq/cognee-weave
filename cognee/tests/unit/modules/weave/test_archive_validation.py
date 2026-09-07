import io
import os
import tarfile
import zipfile
from pathlib import Path

import pytest


def _zip(path: Path, entries: list[tuple[str, bytes]]) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in entries:
            archive.writestr(name, content)
    return path


@pytest.mark.parametrize(
    "name",
    ["../secret", "repo/../../secret", "/etc/passwd", "C:/Windows/system.ini", "repo\\evil"],
)
def test_archive_rejects_traversal_and_absolute_paths(tmp_path, name):
    from cognee.modules.weave.archive import UnsafeArchiveError, validated_archive

    path = _zip(tmp_path / "bad.zip", [(name, b"nope")])

    with pytest.raises(UnsafeArchiveError):
        with validated_archive(path):
            pass


def test_archive_rejects_links_and_devices(tmp_path):
    from cognee.modules.weave.archive import UnsafeArchiveError, validated_archive

    path = tmp_path / "bad.tar"
    with tarfile.open(path, "w") as archive:
        link = tarfile.TarInfo("repo/link")
        link.type = tarfile.SYMTYPE
        link.linkname = "/etc/passwd"
        archive.addfile(link)

        device = tarfile.TarInfo("repo/device")
        device.type = tarfile.CHRTYPE
        archive.addfile(device)

    with pytest.raises(UnsafeArchiveError):
        with validated_archive(path):
            pass


def test_archive_rejects_duplicate_paths(tmp_path):
    from cognee.modules.weave.archive import UnsafeArchiveError, validated_archive

    path = tmp_path / "duplicate.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("repo/app.py", b"one")
        archive.writestr("repo/app.py", b"two")

    with pytest.raises(UnsafeArchiveError, match="duplicate"):
        with validated_archive(path):
            pass


def test_archive_enforces_file_count_and_uncompressed_size(tmp_path):
    from cognee.modules.weave.archive import ArchiveLimits, UnsafeArchiveError, validated_archive

    path = _zip(
        tmp_path / "large.zip",
        [("repo/a.py", b"a" * 32), ("repo/b.py", b"b" * 32)],
    )

    with pytest.raises(UnsafeArchiveError):
        with validated_archive(path, ArchiveLimits(max_files=1)):
            pass

    with pytest.raises(UnsafeArchiveError):
        with validated_archive(path, ArchiveLimits(max_uncompressed_bytes=48)):
            pass


def test_archive_rejects_extreme_compression_ratio(tmp_path):
    from cognee.modules.weave.archive import ArchiveLimits, UnsafeArchiveError, validated_archive

    path = _zip(tmp_path / "bomb.zip", [("repo/zeros.bin", b"0" * 100_000)])

    with pytest.raises(UnsafeArchiveError, match="compression"):
        with validated_archive(path, ArchiveLimits(max_compression_ratio=2)):
            pass


def test_archive_extracts_to_private_directory_and_always_cleans_up(tmp_path):
    from cognee.modules.weave.archive import validated_archive

    path = _zip(tmp_path / "good.zip", [("repo/app.py", b"print('ok')")])
    extracted = None

    with pytest.raises(RuntimeError):
        with validated_archive(path) as repository:
            extracted = repository.parent
            assert repository.name == "repo"
            assert (repository / "app.py").read_bytes() == b"print('ok')"
            assert os.stat(extracted).st_mode & 0o077 == 0
            raise RuntimeError("pipeline failed")

    assert extracted is not None
    assert not extracted.exists()
