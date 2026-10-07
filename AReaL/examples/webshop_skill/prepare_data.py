# SPDX-License-Identifier: MIT

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

DOWNLOADS = {
    "small": {
        "items_shuffle_1000.json": "https://drive.google.com/uc?id=1EgHdxQ_YxqIQlvvq5iKlCrkEKR6-j0Ib",
        "items_ins_v2_1000.json": "https://drive.google.com/uc?id=1IduG0xl544V_A_jv3tHXC0kyFi7PnyBu",
    },
    "full": {
        "items_shuffle.json": "https://drive.google.com/uc?id=1A2whVgOO0euk5O13n2iYDM0bQRkkRduB",
        "items_ins_v2.json": "https://drive.google.com/uc?id=1s2j6NgHljiZzQNL3veZaAiyW_qDEgBNi",
    },
}
HUMAN_ATTRIBUTES = {
    "items_human_ins.json": "https://drive.google.com/uc?id=14Kb5SPBk_jfdLZ_CDBNitW98QLDlKR5O"
}
DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _download_files(data_root: Path, mode: str) -> None:
    for filename, url in {**DOWNLOADS[mode], **HUMAN_ATTRIBUTES}.items():
        output = data_root / filename
        if output.exists() and output.stat().st_size > 0:
            print(f"[download] reusing {output}", flush=True)
            continue
        partial = output.with_suffix(output.suffix + ".part")
        print(f"[download:google-drive] {filename} -> {partial.name}", flush=True)
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError(
                "install gdown in the shared benchmark environment"
            ) from exc
        result = gdown.download(
            url,
            str(partial),
            quiet=False,
            resume=partial.exists() and partial.stat().st_size > 0,
        )
        if not result:
            raise RuntimeError(f"download failed: {url}")
        if not partial.exists() or partial.stat().st_size <= 0:
            raise RuntimeError(f"download produced an empty file: {filename}")
        partial.replace(output)


def _paths(data_root: Path, mode: str) -> tuple[Path, Path, Path, Path, Path]:
    suffix = "_1000" if mode == "small" else ""
    products = data_root / f"items_shuffle{suffix}.json"
    attributes = data_root / f"items_ins_v2{suffix}.json"
    human_attributes = data_root / "items_human_ins.json"
    resource_dir = (
        data_root
        / "search_engine"
        / ("resources_1k" if mode == "small" else "resources")
    )
    index_dir = (
        data_root / "search_engine" / ("indexes_1k" if mode == "small" else "indexes")
    )
    return products, attributes, human_attributes, resource_dir, index_dir


def _load_products(
    *,
    repo_root: Path,
    products_file: Path,
    attributes_file: Path,
    human_attributes_file: Path,
    mode: str,
) -> list[dict[str, Any]]:
    # This script also runs directly from the repository root, outside an
    # AReaL launcher. Make the shared runtime helper available in that case.
    areal_root = str(Path(__file__).resolve().parents[2])
    if areal_root not in sys.path:
        sys.path.insert(0, areal_root)
    from examples.webshop_skill.native_runtime import lucene_searcher_class

    # Loading Pyserini first can crash while its JVM is active and the engine
    # subsequently imports native Torch, spaCy, or ONNX dependencies.
    lucene_searcher_class()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from web_agent_site import utils as webshop_utils

    webshop_utils.DEFAULT_FILE_PATH = str(products_file)
    webshop_utils.DEFAULT_ATTR_PATH = str(attributes_file)
    webshop_utils.HUMAN_ATTR_PATH = str(human_attributes_file)
    from web_agent_site.engine import engine

    engine.DEFAULT_FILE_PATH = str(products_file)
    engine.DEFAULT_ATTR_PATH = str(attributes_file)
    engine.HUMAN_ATTR_PATH = str(human_attributes_file)
    products, *_ = engine.load_products(
        filepath=str(products_file),
        num_products=1000 if mode == "small" else None,
        human_goals=False,
    )
    return products


def _write_documents(products: list[dict[str, Any]], resource_dir: Path) -> Path:
    resource_dir.mkdir(parents=True, exist_ok=True)
    output = resource_dir / "documents.jsonl"
    temp = output.with_suffix(".jsonl.tmp")
    print(f"[convert] writing {len(products)} products to {output}", flush=True)
    with temp.open("w", encoding="utf-8") as handle:
        for product in products:
            option_texts = []
            for option_name, values in (product.get("options") or {}).items():
                option_texts.append(f"{option_name}: {', '.join(values)}")
            bullet_points = product.get("BulletPoints") or []
            contents = " ".join(
                [
                    str(product.get("Title") or ""),
                    str(product.get("Description") or ""),
                    str(bullet_points[0] if bullet_points else ""),
                    ", and ".join(option_texts),
                ]
            ).lower()
            handle.write(
                json.dumps(
                    {
                        "id": product["asin"],
                        "contents": contents,
                        "product": product,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    temp.replace(output)
    return output


def _build_index(
    resource_dir: Path, index_dir: Path, *, threads: int, force: bool
) -> None:
    marker = index_dir.parent / f"{index_dir.name}.complete.json"
    if index_dir.exists() and any(index_dir.iterdir()) and marker.is_file():
        if not force:
            print(f"[index] reusing complete index {index_dir}", flush=True)
            return
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    building = index_dir.with_name(index_dir.name + ".building")
    if building.exists():
        shutil.rmtree(building)
    if force and marker.exists():
        marker.unlink()
    cmd = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(resource_dir),
        "--index",
        str(building),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(max(1, threads)),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]
    print(f"[index] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)
    if not building.exists() or not any(building.iterdir()):
        raise RuntimeError(f"Pyserini produced an empty index: {building}")
    if index_dir.exists():
        shutil.rmtree(index_dir)
    building.replace(index_dir)
    marker.write_text(
        json.dumps(
            {
                "status": "complete",
                "index": str(index_dir),
                "documents": str(resource_dir / "documents.jsonl"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and index WebShop data outside the source tree."
    )
    parser.add_argument("--mode", choices=("small", "full"), default="full")
    parser.add_argument(
        "--download-source",
        choices=("google-drive",),
        default="google-drive",
        help="Data host: the official WebShop Google Drive files.",
    )
    parser.add_argument(
        "--repo-root",
        default=os.environ.get(
            "WEBSHOP_REPO_ROOT",
            os.environ.get(
                "WEBSHOP_ROOT", str(DEFAULT_PROJECT_ROOT / ".benchmark-runtime/webshop")
            ),
        ),
    )
    parser.add_argument(
        "--data-root",
        default=os.environ.get(
            "WEBSHOP_DATA_ROOT",
            str(DEFAULT_PROJECT_ROOT / "data/webshop"),
        ),
    )
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--force-index",
        action="store_true",
        help="Delete and rebuild an existing Lucene index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    data_root = Path(args.data_root).expanduser().resolve()
    if not (repo_root / "web_agent_site").is_dir():
        raise FileNotFoundError(f"WebShop repository is missing: {repo_root}")
    data_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_download:
        _download_files(data_root, args.mode)
    products, attributes, human_attributes, resource_dir, index_dir = _paths(
        data_root, args.mode
    )
    for path in (products, attributes, human_attributes):
        if not path.exists():
            raise FileNotFoundError(f"required data file is missing: {path}")
    product_rows = _load_products(
        repo_root=repo_root,
        products_file=products,
        attributes_file=attributes,
        human_attributes_file=human_attributes,
        mode=args.mode,
    )
    _write_documents(product_rows, resource_dir)
    _build_index(
        resource_dir,
        index_dir,
        threads=args.threads,
        force=args.force_index,
    )
    print(
        json.dumps(
            {
                "mode": args.mode,
                "products_file": str(products),
                "attributes_file": str(attributes),
                "human_attributes_file": str(human_attributes),
                "search_index": str(index_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
