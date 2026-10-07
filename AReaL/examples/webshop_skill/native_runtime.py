# SPDX-License-Identifier: MIT
"""Load the simulator's native dependencies before Pyserini starts its JVM."""

from functools import lru_cache
from importlib import import_module


@lru_cache(maxsize=1)
def lucene_searcher_class():
    # Load native CPU libraries before Java to avoid initialization conflicts; do not initialize CUDA.
    for name in ("torch", "transformers.modeling_utils", "spacy", "onnxruntime"):
        print(f"[webshop-native] loading {name} before JVM", flush=True)
        import_module(name)
    print("[webshop-native] loading Lucene/JVM", flush=True)
    searcher = import_module("pyserini.search.lucene").LuceneSearcher
    print("[webshop-native] Lucene/JVM ready", flush=True)
    return searcher
