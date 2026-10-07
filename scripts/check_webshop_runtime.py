"""Check the real Lucene import path and optionally query a local search index."""
import argparse
from pathlib import Path
import sys


def load_searcher():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "AReaL"))
    from examples.webshop_skill.native_runtime import lucene_searcher_class
    return lucene_searcher_class()


def check(search_index=None):
    try:
        searcher_class = load_searcher()
    except ModuleNotFoundError as exc:
        package = {"sklearn": "scikit-learn", "jnius": "pyjnius"}.get(
            (exc.name or "").split(".")[0], exc.name
        )
        raise RuntimeError(
            f"WebShop Lucene dependency missing: {package}. "
            "Install requirements-env.txt in the training Python environment."
        ) from exc
    if search_index is not None:
        path = Path(search_index)
        if not path.is_dir():
            raise RuntimeError(f"WebShop search index not found: {path}")
        searcher = searcher_class(str(path))
        try:
            if not searcher.search("red running shoes", k=1):
                raise RuntimeError("WebShop Lucene smoke query returned no hits")
        finally:
            close = getattr(searcher, "close", None)
            if close is not None:
                close()
    print("WebShop Lucene runtime check passed", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-index", type=Path)
    check(parser.parse_args().search_index)
