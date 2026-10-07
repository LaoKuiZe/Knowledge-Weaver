#!/usr/bin/env python3
"""Install the pinned official WebShop text runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import site
import tarfile
import tempfile
import urllib.request


COMMIT = "64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd"
ARCHIVE_URL = f"https://codeload.github.com/princeton-nlp/WebShop/tar.gz/{COMMIT}"
ARCHIVE_SHA256 = "2e3e671ad76ee5dd1b143f1a060f91386b9699602bee794259b5de79ebfaea8a"
GOAL_SHA256 = "9703c8583244e2041182eb14856186b998145ab16ce2ba9aef8cb680877c8ed5"
DEFAULT_TARGET = Path(__file__).resolve().parents[1] / ".benchmark-runtime" / "webshop"
MANIFEST = ".knowledgeweaver-runtime.json"
IDENTITY = {"commit": COMMIT, "archive_sha256": ARCHIVE_SHA256, "patch_version": 1}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_installation(target: Path) -> None:
    """Never silently reuse a partial, changed, or differently pinned runtime."""
    try:
        manifest = json.loads((target / MANIFEST).read_text())
        if not isinstance(manifest, dict):
            raise ValueError("runtime manifest must be an object")
        if any(manifest.get(key) != value for key, value in IDENTITY.items()):
            raise ValueError("source version does not match")
        files = manifest.get("files", {})
        required = {"LICENSE.md", "web_agent_site/__init__.py", "web_agent_site/engine/goal.py"}
        if not required.issubset(files):
            raise ValueError("runtime manifest is incomplete")
        for name, expected in files.items():
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name:
                raise ValueError("invalid path in runtime manifest")
            path = target / name
            if path.is_symlink() or not path.resolve().is_relative_to(target.resolve()):
                raise ValueError(f"unexpected symlink: {name}")
            if sha256(path) != expected:
                raise ValueError(f"runtime file changed: {name}")
    except (OSError, ValueError, TypeError) as exc:
        raise RuntimeError(
            f"Existing WebShop runtime at {target} failed verification ({exc}). "
            "Move it aside and rerun setup; its contents were not overwritten."
        ) from exc


def extract_runtime(archive: Path, target: Path) -> None:
    """Copy only the text environment and upstream license, with no tar extraction."""
    prefix = f"WebShop-{COMMIT}"
    seen: set[str] = set()
    with tarfile.open(archive, "r:gz") as source:
        for member in source:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or "\\" in member.name:
                raise RuntimeError(f"Unsafe archive path: {member.name}")
            if not path.parts or path.parts[0] != prefix:
                raise RuntimeError(f"Unexpected archive root: {member.name}")
            relative = PurePosixPath(*path.parts[1:])
            if relative.as_posix() == "web_agent_site/envs/chromedriver":
                continue  # Browser binary is unused by text experiments.
            if relative.as_posix() != "LICENSE.md" and (
                not relative.parts or relative.parts[0] != "web_agent_site"
            ):
                continue
            if member.isdir():
                continue
            if not member.isfile() or relative.as_posix() in seen:
                raise RuntimeError(f"Unsupported archive entry: {member.name}")
            seen.add(relative.as_posix())
            output = target / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            with source.extractfile(member) as incoming, output.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)

    goal = target / "web_agent_site" / "engine" / "goal.py"
    if not goal.is_file() or sha256(goal) != GOAL_SHA256:
        raise RuntimeError("Pinned WebShop goal.py does not match the expected source")
    # Preserve the sole experiment-specific change to the upstream environment:
    # a configurable spaCy model path, retaining the upstream small-model default.
    text = goal.read_text()
    text = text.replace("import itertools\n", "import itertools\nimport os\n", 1)
    text = text.replace(
        'nlp = spacy.load("en_core_web_sm")',
        'nlp = spacy.load(os.environ.get("WEBSHOP_SPACY_MODEL_PATH", "en_core_web_sm"))',
        1,
    )
    goal.write_text(text)
    files = {
        file.relative_to(target).as_posix(): sha256(file)
        for file in sorted(target.rglob("*"))
        if file.is_file()
    }
    (target / MANIFEST).write_text(json.dumps({**IDENTITY, "files": files}, indent=2) + "\n")
    verify_installation(target)


def install(target: Path, archive: Path | None = None) -> bool:
    target = target.expanduser().absolute()
    if target.is_symlink():
        raise RuntimeError(f"Refusing a symlink installation target: {target}")
    if target.exists():
        verify_installation(target)
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".webshop-install-", dir=target.parent) as scratch:
        scratch = Path(scratch)
        if archive is None:
            archive = scratch / "webshop.tar.gz"
            request = urllib.request.Request(ARCHIVE_URL, headers={"User-Agent": "Knowledge-Weaver-setup"})
            with urllib.request.urlopen(request, timeout=60) as incoming, archive.open("wb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
        if sha256(archive) != ARCHIVE_SHA256:
            raise RuntimeError("WebShop archive SHA256 mismatch; installation was not changed")
        staged = scratch / "runtime"
        staged.mkdir()
        extract_runtime(archive, staged)
        if target.exists():
            raise RuntimeError(f"Installation target appeared during setup: {target}")
        staged.rename(target)
    return True


def register_pth(target: Path) -> None:
    value = str(target.expanduser().resolve())
    if "\n" in value or "\r" in value:
        raise RuntimeError("WebShop installation path cannot contain a newline")
    # A fixed filename, so registering again replaces any earlier path.
    registration = Path(site.getsitepackages()[0]) / "webshop_repo.pth"
    registration.write_text(value + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET, help="runtime directory")
    parser.add_argument("--archive", type=Path, help="use a cached archive (SHA256 is still checked)")
    parser.add_argument("--register-pth", action="store_true", help="register in this Python environment")
    args = parser.parse_args()
    try:
        changed = install(args.target, args.archive)
        if args.register_pth:
            register_pth(args.target)
    except (OSError, RuntimeError, tarfile.TarError) as exc:
        parser.exit(1, f"WebShop setup failed: {exc}\n")
    print(f"WebShop {'installed' if changed else 'verified'}: {args.target.resolve()} ({COMMIT})")


if __name__ == "__main__":
    main()
