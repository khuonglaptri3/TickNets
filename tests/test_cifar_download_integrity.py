"""Corrupt caches, official-file validation, fallback download and safe extraction."""
import hashlib
import io
import tarfile
import urllib.error

import pytest

from scripts import download_cifar as download


@pytest.mark.parametrize("dataset", ("cifar10", "cifar100"))
@pytest.mark.parametrize("cache", ("missing", "unrelated", "corrupt", "valid"))
def test_cache_is_verified_or_repaired(tmp_path, monkeypatch, dataset, cache):
    folder = tmp_path / download.DATASET_METADATA[dataset]["extracted_dir"]
    expected = {"train": b"training", "test": b"testing", "meta": b"labels"}
    monkeypatch.setitem(download.EXTRACTED_CHECKSUMS, dataset,
                        {k: hashlib.md5(v).hexdigest() for k, v in expected.items()})
    if cache != "missing":
        folder.mkdir()
        if cache == "unrelated":
            (folder / "unrelated.txt").write_text("unrelated")
        else:
            for name, value in expected.items():
                (folder / name).write_bytes(value)
            if cache == "corrupt":
                (folder / "train").write_bytes(b"bad")
    attempts = []
    monkeypatch.setattr(download, "download_file_with_fallback", lambda *a, **kw: attempts.append(a))
    def extract(*args):
        folder.mkdir(exist_ok=True)
        for name, value in expected.items():
            (folder / name).write_bytes(value)
    monkeypatch.setattr(download, "extract_tar_gz", extract)
    result = download.ensure_cifar_dataset(dataset, tmp_path)
    assert result["status"] == ("already_present" if cache == "valid" else "downloaded_and_extracted")
    assert len(attempts) == (0 if cache == "valid" else 1)


def test_post_extraction_validation_failure_is_not_success(tmp_path, monkeypatch):
    monkeypatch.setattr(download, "download_file_with_fallback", lambda *a, **kw: None)
    monkeypatch.setattr(download, "extract_tar_gz", lambda *a: None)
    with pytest.raises(ValueError, match="MD5"):
        download.ensure_cifar_dataset("cifar10", tmp_path)


@pytest.mark.parametrize("member_name,member_type", [("../escape", tarfile.REGTYPE),
                                                     ("/escape", tarfile.REGTYPE),
                                                     ("link", tarfile.SYMTYPE),
                                                     ("link", tarfile.LNKTYPE)])
def test_unsafe_archive_is_rejected_before_extraction(tmp_path, member_name, member_type):
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo(member_name)
        member.type = member_type
        member.linkname = "../escape"
        handle.addfile(member)
    destination = tmp_path / "destination"
    with pytest.raises(ValueError, match="Unsafe"):
        download.extract_tar_gz(archive, destination)
    assert not destination.exists()


def test_download_fallback_and_md5_rejection_leave_no_partial_file(tmp_path, monkeypatch):
    calls = []
    class Response(io.BytesIO):
        headers = {"content-length": "4"}
    class Opener:
        def open(self, request, timeout):
            calls.append(request.full_url)
            if "primary" in request.full_url:
                raise urllib.error.URLError("offline primary")
            return Response(b"good")
    monkeypatch.setattr(download.urllib.request, "build_opener", lambda *a: Opener())
    target = tmp_path / "data.tar.gz"
    download.download_file_with_fallback(["https://primary", "https://fallback"], target,
                                         hashlib.md5(b"good").hexdigest())
    assert target.read_bytes() == b"good"
    assert "https://fallback" in calls
    target.unlink()
    with pytest.raises(RuntimeError):
        download.download_file_with_fallback(["https://fallback"], target, "0" * 32)
    assert not target.exists() and not target.with_suffix(".tmp").exists()
