"""Fetch fixed-commit author archives locally; verify SHA-256 before extracting.

This script never installs dependencies or executes downloaded author code.
Third-party source, examples, and archives remain excluded by .gitignore.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
PROJECTS = ("stripping_discount_curve", "shrinking_term_structure", "sharpe_ratio_inference")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def specification(name):
    project = ROOT / name
    manifest = json.loads((project / "source_manifest.json").read_text())
    if name == "stripping_discount_curve":
        prefix = "original/author/"
        entries = {r["path"][len(prefix):]: r["sha256"]
                   for r in manifest["files"] if r["path"].startswith(prefix)}
        archive_row = next(r for r in manifest["files"] if r["path"] == "original/author_repository.zip")
        return dict(project=project, target=project / "original" / "author", entries=entries,
                    archive=project / "original" / "author_repository.zip",
                    archive_sha256=archive_row["sha256"], url=manifest["archive_url"])
    entries = manifest["files"]
    if isinstance(entries, list):
        entries = {r["path"]: r["sha256"] for r in entries}
    return dict(project=project, target=project / "source", entries=entries,
                archive=project / "source_archive.zip", archive_sha256=manifest["archive_sha256"],
                url=manifest.get("archive_url", manifest.get("download_url")))


def verify_files(spec):
    for relative, expected in spec["entries"].items():
        path = spec["target"] / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing downloaded author file: {path.relative_to(ROOT)}")
        if sha256(path) != expected:
            raise ValueError(f"Author file hash mismatch: {path.relative_to(ROOT)}")
    if not spec["archive"].is_file() or sha256(spec["archive"]) != spec["archive_sha256"]:
        raise ValueError(f"Missing or mismatched archive: {spec['archive'].relative_to(ROOT)}")


def extract_verified(archive, target, entries):
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing source directory: {target.relative_to(ROOT)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="source-stage-", dir=target.parent) as staging:
        staged = Path(staging) / "source"
        staged.mkdir()
        seen = set()
        archive_roots = set()
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                parts = PurePosixPath(info.filename).parts
                if not parts or PurePosixPath(info.filename).is_absolute() or ".." in parts:
                    raise ValueError("Unsafe archive member")
                archive_roots.add(parts[0])
                if stat.S_ISLNK(info.external_attr >> 16):
                    raise ValueError("Symlinks are not permitted in author archives")
                if info.is_dir():
                    continue
                relative = PurePosixPath(*parts[1:]).as_posix()
                if relative not in entries or relative in seen:
                    raise ValueError(f"Unexpected or duplicate archive member: {relative}")
                # Fixed manifest verifies every file, in addition to the full ZIP hash.
                payload = z.read(info)
                if hashlib.sha256(payload).hexdigest() != entries[relative]:
                    raise ValueError(f"Archive member hash mismatch: {relative}")
                path = staged / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                seen.add(relative)
        if len(archive_roots) != 1 or seen != set(entries):
            raise ValueError("Archive contents do not match the fixed manifest")
        staged.replace(target)


def prepare(name, archive_dir=None, verify_only=False):
    spec = specification(name)
    if verify_only or spec["target"].exists():
        verify_files(spec)
        print(f"{name}: verified {len(spec['entries'])} original files and archive", flush=True)
        return
    spec["archive"].parent.mkdir(parents=True, exist_ok=True)
    if not spec["archive"].exists():
        with tempfile.NamedTemporaryFile(prefix="archive-", suffix=".zip", dir=spec["archive"].parent, delete=False) as handle:
            temp = Path(handle.name)
        try:
            if archive_dir is not None:
                local = Path(archive_dir) / f"{name}.zip"
                print(f"{name}: using supplied fixed-commit archive", flush=True)
                shutil.copyfile(local, temp)
            else:
                request = urllib.request.Request(spec["url"], headers={"User-Agent": "finance-econometrics-replications/1.0"})
                print(f"{name}: downloading public fixed-commit archive", flush=True)
                with urllib.request.urlopen(request, timeout=120) as response, temp.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if sha256(temp) != spec["archive_sha256"]:
                raise ValueError(f"Archive SHA-256 mismatch: {name}")
            temp.replace(spec["archive"])
        finally:
            temp.unlink(missing_ok=True)
    if sha256(spec["archive"]) != spec["archive_sha256"]:
        raise ValueError(f"Archive SHA-256 mismatch: {name}")
    extract_verified(spec["archive"], spec["target"], spec["entries"])
    verify_files(spec)
    print(f"{name}: downloaded and verified {len(spec['entries'])} original files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", choices=PROJECTS, help="Prepare only one experiment; default prepares all three")
    parser.add_argument("--verify-only", action="store_true", help="Check existing downloads without network access")
    parser.add_argument("--archive-dir", type=Path, help="Offline archive directory containing PROJECT_NAME.zip files")
    args = parser.parse_args()
    for name in [args.project] if args.project else PROJECTS:
        prepare(name, args.archive_dir, args.verify_only)


if __name__ == "__main__":
    main()
