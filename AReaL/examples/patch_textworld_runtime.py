#!/usr/bin/env python3
# SPDX-License-Identifier: MIT

"""Apply the ALFWorld/TextWorld concurrency fixes to the active environment."""

from __future__ import annotations

import argparse
import importlib.util
import py_compile
import re
import shutil
from pathlib import Path

FAST_MARKER = "AREAL_FAST_DOWNWARD_MEMFD_LIFETIME_V2"
TEXTWORLD_MARKER = "AREAL_TEXTWORLD_FAST_DOWNWARD_CLOSE_FIX"

FAST_FUNCTIONS = r'''def load_lib():
    """Load an isolated Fast Downward library from an anonymous memfd."""

    import tempfile

    # AREAL_FAST_DOWNWARD_MEMFD_LIFETIME_V2: keep each memfd alive until
    # dlclose, and never reuse a dlopen pathname. glibc caches loaded names;
    # /proc/self/fd/N alone aliases environments when N is reused. A unique
    # symlink also prevents aliasing libraries retained after dlclose (NODELETE).
    # The symlink is tiny; the library itself remains in anonymous memory.
    if not hasattr(os, "memfd_create"):
        raise RuntimeError("Fast Downward isolation requires os.memfd_create")
    fd = os.memfd_create("areal-libdownward", flags=os.MFD_CLOEXEC)
    try:
        with open(DOWNWARD_LIB_PATH, "rb") as source:
            with os.fdopen(os.dup(fd), "wb") as target:
                shutil.copyfileobj(source, target)
        os.lseek(fd, 0, os.SEEK_SET)
        with tempfile.TemporaryDirectory(prefix="areal-downward-") as directory:
            library_path = os.path.join(directory, "libdownward.so")
            os.symlink(f"/proc/self/fd/{fd}", library_path)
            downward_lib = cdll.LoadLibrary(library_path)
    except BaseException:
        os.close(fd)
        raise
    downward_lib._areal_memfd = fd
    downward_lib._areal_closed = False

    downward_lib.load_sas.argtypes = [c_char_p]
    downward_lib.load_sas.restype = None
    downward_lib.load_sas_replan.argtypes = [c_char_p]
    downward_lib.load_sas_replan.restype = None
    downward_lib.cleanup.argtypes = []
    downward_lib.cleanup.restype = None
    downward_lib.get_applicable_operators_count.argtypes = []
    downward_lib.get_applicable_operators_count.restype = int
    downward_lib.get_applicable_operators.argtypes = [POINTER(Operator)]
    downward_lib.get_applicable_operators.restype = None
    downward_lib.get_state_size.argtypes = []
    downward_lib.get_state_size.restype = int
    downward_lib.get_state.argtypes = [POINTER(Atom)]
    downward_lib.get_state.restype = None
    downward_lib.apply_operator.argtypes = [c_int, POINTER(Atom)]
    downward_lib.apply_operator.restype = int
    downward_lib.check_goal.argtypes = []
    downward_lib.check_goal.restype = bool
    downward_lib.solve.argtypes = [c_bool]
    downward_lib.solve.restype = bool
    downward_lib.solve_sas.argtypes = [c_char_p, c_bool]
    downward_lib.solve_sas.restype = bool
    downward_lib.replan.argtypes = [c_bool]
    downward_lib.replan.restype = bool
    downward_lib.get_last_plan_length.argtypes = []
    downward_lib.get_last_plan_length.restype = int
    downward_lib.get_last_plan.argtypes = [POINTER(Operator)]
    downward_lib.get_last_plan.restype = None
    downward_lib.check_solution.argtypes = [c_int, POINTER(Operator)]
    downward_lib.check_solution.restype = bool
    return downward_lib


def close_lib(downward_lib):
    if downward_lib.__dict__.get("_areal_closed", False):
        return
    downward_lib._areal_closed = True
    handle = downward_lib._handle
    fd = downward_lib.__dict__.get("_areal_memfd")
    try:
        downward_lib.cleanup()
    finally:
        try:
            if dlclose_func(handle):
                raise RuntimeError("Fast Downward dlclose failed")
        finally:
            downward_lib._handle = 0
            if fd is not None:
                os.close(fd)
                downward_lib._areal_memfd = None
'''

TEXTWORLD_CLOSE = r"""
    def close(self) -> None:
        # AREAL_TEXTWORLD_FAST_DOWNWARD_CLOSE_FIX: PddlEnv owns a private
        # Fast Downward library copy. Release it when ALFWorld closes the env.
        downward_lib = getattr(self, "downward_lib", None)
        if downward_lib is None:
            return
        self.downward_lib = None
        self._pddl_state = None
        try:
            fast_downward.close_lib(downward_lib)
        finally:
            del downward_lib
            gc.collect()
"""


def _module_path(name: str) -> Path:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        raise RuntimeError(f"required module is not installed: {name}")
    return Path(spec.origin).resolve()


def _backup(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".areal-backup")
    if not backup.exists():
        shutil.copy2(path, backup)


def patch_fast_downward(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if FAST_MARKER in text:
        return False
    pattern = re.compile(
        r"def load_lib\(\):.*?\n\ndef close_lib\(downward_lib\):.*?(?=\n\ndef pddl2sas)",
        re.DOTALL,
    )
    updated, count = pattern.subn(FAST_FUNCTIONS.rstrip(), text, count=1)
    if count != 1:
        raise RuntimeError(f"unsupported fast_downward interface layout: {path}")
    _backup(path)
    path.write_text(updated, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    return True


def patch_textworld(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if TEXTWORLD_MARKER in text:
        return False
    import_anchor = "import json\n"
    init_anchor = "        self.downward_lib = fast_downward.load_lib()\n"
    if import_anchor not in text or init_anchor not in text:
        raise RuntimeError(f"unsupported TextWorld PDDL layout: {path}")
    updated = text.replace(import_anchor, "import gc\nimport json\n", 1)
    updated = updated.replace(init_anchor, init_anchor + TEXTWORLD_CLOSE, 1)
    _backup(path)
    path.write_text(updated, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Only verify markers.")
    args = parser.parse_args()
    fast_path = _module_path("fast_downward.interface")
    textworld_path = _module_path("textworld.envs.pddl.pddl")
    if args.check:
        missing = []
        if FAST_MARKER not in fast_path.read_text(encoding="utf-8"):
            missing.append(str(fast_path))
        if TEXTWORLD_MARKER not in textworld_path.read_text(encoding="utf-8"):
            missing.append(str(textworld_path))
        if missing:
            raise RuntimeError("missing runtime patches: " + ", ".join(missing))
        print("TextWorld/Fast Downward runtime patches: ok")
        return
    changed = [patch_fast_downward(fast_path), patch_textworld(textworld_path)]
    print(f"TextWorld/Fast Downward runtime patches: ok (changed={sum(changed)})")


if __name__ == "__main__":
    main()
