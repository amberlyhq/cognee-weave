import shutil
import stat
import os
import tarfile
import tempfile
import zipfile
from contextlib import contextmanager
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Iterator


class UnsafeArchiveError(ValueError):
    """The uploaded archive cannot be extracted safely."""


@dataclass(frozen=True)
class ArchiveLimits:
    max_archive_bytes: int = 100 * 1024 * 1024
    max_files: int = 20_000
    max_file_bytes: int = 25 * 1024 * 1024
    max_uncompressed_bytes: int = 500 * 1024 * 1024
    max_compression_ratio: float = 200.0


def _safe_relative_path(name: str) -> PurePosixPath:
    if not name or "\\" in name or "\x00" in name:
        raise UnsafeArchiveError("Archive contains an invalid path")
    path = PurePosixPath(name)
    windows_path = PureWindowsPath(name)
    if path.is_absolute() or windows_path.is_absolute() or windows_path.drive:
        raise UnsafeArchiveError("Archive contains an absolute path")
    if any(part in ("", ".", "..") for part in path.parts):
        raise UnsafeArchiveError("Archive contains a traversal path")
    return path


def _check_budget(
    *,
    file_count: int,
    file_size: int,
    total_size: int,
    compressed_size: int,
    limits: ArchiveLimits,
) -> None:
    if file_count > limits.max_files:
        raise UnsafeArchiveError("Archive exceeds the file-count limit")
    if file_size > limits.max_file_bytes:
        raise UnsafeArchiveError("Archive member exceeds the file-size limit")
    if total_size > limits.max_uncompressed_bytes:
        raise UnsafeArchiveError("Archive exceeds the uncompressed-size limit")
    if file_size and compressed_size == 0:
        raise UnsafeArchiveError("Archive member has an invalid compression size")
    if compressed_size and file_size / compressed_size > limits.max_compression_ratio:
        raise UnsafeArchiveError("Archive exceeds the compression-ratio limit")


def _copy_member(source: BinaryIO, destination: Path, expected_size: int) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    written = 0
    with destination.open("xb") as output:
        while True:
            chunk = source.read(min(1024 * 1024, expected_size - written + 1))
            if not chunk:
                break
            written += len(chunk)
            if written > expected_size:
                raise UnsafeArchiveError("Archive member expanded beyond its declared size")
            output.write(chunk)
    destination.chmod(0o600)
    if written != expected_size:
        raise UnsafeArchiveError("Archive member size does not match its declaration")


def _extract_zip(archive_path: Path, destination: Path, limits: ArchiveLimits) -> None:
    seen: set[PurePosixPath] = set()
    file_count = 0
    total_size = 0
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            relative = _safe_relative_path(member.filename.rstrip("/"))
            if relative in seen:
                raise UnsafeArchiveError("Archive contains a duplicate path")
            seen.add(relative)

            unix_mode = member.external_attr >> 16
            if stat.S_ISLNK(unix_mode):
                raise UnsafeArchiveError("Archive links are not allowed")
            if member.flag_bits & 0x1:
                raise UnsafeArchiveError("Encrypted archive members are not allowed")

            target = destination.joinpath(*relative.parts)
            if member.is_dir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue

            file_count += 1
            total_size += member.file_size
            _check_budget(
                file_count=file_count,
                file_size=member.file_size,
                total_size=total_size,
                compressed_size=member.compress_size,
                limits=limits,
            )
            with archive.open(member, "r") as source:
                _copy_member(source, target, member.file_size)


def _extract_tar(archive_path: Path, destination: Path, limits: ArchiveLimits) -> None:
    seen: set[PurePosixPath] = set()
    file_count = 0
    total_size = 0
    archive_size = max(archive_path.stat().st_size, 1)
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            relative = _safe_relative_path(member.name.rstrip("/"))
            if relative in seen:
                raise UnsafeArchiveError("Archive contains a duplicate path")
            seen.add(relative)

            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise UnsafeArchiveError("Archive links and devices are not allowed")

            file_count += 1
            total_size += member.size
            _check_budget(
                file_count=file_count,
                file_size=member.size,
                total_size=total_size,
                compressed_size=archive_size,
                limits=limits,
            )
            if total_size / archive_size > limits.max_compression_ratio:
                raise UnsafeArchiveError("Archive exceeds the compression-ratio limit")
            source = archive.extractfile(member)
            if source is None:
                raise UnsafeArchiveError("Archive member cannot be read")
            with source:
                _copy_member(source, target, member.size)


def _repository_root(destination: Path) -> Path:
    entries = list(destination.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return destination


@contextmanager
def validated_archive(
    archive_path: Path | str,
    limits: ArchiveLimits = ArchiveLimits(),
) -> Iterator[Path]:
    """Extract a bounded zip/tar archive into a private, temporary directory."""

    archive_path = Path(archive_path)
    if not archive_path.is_file():
        raise UnsafeArchiveError("Archive does not exist")
    if archive_path.stat().st_size > limits.max_archive_bytes:
        raise UnsafeArchiveError("Archive exceeds the upload-size limit")

    temporary = Path(tempfile.mkdtemp(prefix="cognee-weave-"))
    temporary.chmod(0o700)
    destination = temporary / "content"
    destination.mkdir(mode=0o700)
    try:
        if zipfile.is_zipfile(archive_path):
            _extract_zip(archive_path, destination, limits)
        elif tarfile.is_tarfile(archive_path):
            _extract_tar(archive_path, destination, limits)
        else:
            raise UnsafeArchiveError("Only zip and tar archives are supported")
        yield _repository_root(destination)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


@asynccontextmanager
async def persisted_upload(upload, limits: ArchiveLimits = ArchiveLimits()) -> Iterator[Path]:
    """Copy an async upload to a bounded private file and remove it afterwards."""

    descriptor, raw_path = tempfile.mkstemp(prefix="cognee-weave-upload-")
    path = Path(raw_path)
    os.chmod(path, 0o600)
    received = 0
    try:
        with os.fdopen(descriptor, "wb") as output:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                received += len(chunk)
                if received > limits.max_archive_bytes:
                    raise UnsafeArchiveError("Archive exceeds the upload-size limit")
                output.write(chunk)
        yield path
    finally:
        path.unlink(missing_ok=True)
