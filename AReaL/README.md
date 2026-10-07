# AReaL training backend

KnowledgeWeaver uses [AReaL](https://github.com/areal-project/AReaL) at upstream
revision
[`8bb3ff8`](https://github.com/areal-project/AReaL/tree/8bb3ff8af164bd589515fd6bef52b9d7c00f0175),
with our ALFWorld/WebShop training workflows and reward extensions.

The complete `areal/` framework package, installation tools, dependency locks, and
framework tests are retained. Unrelated upstream examples, tutorials, promotional
assets, and maintainer configuration are omitted. Tests tied to omitted examples are
excluded or explicitly skipped; framework tests remain.

```text
areal/                         Complete framework package + our modifications
examples/
├── alfworld_skill/             ALFWorld training and knowledge curation
├── webshop_skill/              WebShop training and environment service
├── skill_training/             Shared semantic knowledge-bank utilities
└── patch_textworld_runtime.py  TextWorld runtime compatibility patch
scripts/                       Upstream dependency sync and lock tools
tests/                         Framework tests
```

Start with the [project README](../README.md) for setup, MSR + SR + MI training,
knowledge-bank construction, and evaluation. See the
[upstream documentation](https://areal-project.github.io/AReaL/) for general AReaL usage
and [THIRD_PARTY.md](../THIRD_PARTY.md) for provenance and release scope.

Upstream AReaL files keep their Apache-2.0 [license](LICENSE) and source notices; the
ones we changed, including this README, carry a notice saying so. Files we added carry
an MIT license header.
