# Third-party code

| Component | Source | License |
| --- | --- | --- |
| AReaL | [areal-project/AReaL@8bb3ff8](https://github.com/areal-project/AReaL/tree/8bb3ff8af164bd589515fd6bef52b9d7c00f0175) | Apache-2.0 ([license](AReaL/LICENSE)) |
| ALFWorld | [alfworld/alfworld@4490f25](https://github.com/alfworld/alfworld/tree/4490f259df671a36e7357a831d3b6c642abd4c25) | MIT ([notice](eval_data/alfworld/LICENSE)) |
| WebShop | [princeton-nlp/WebShop@64fa2a5](https://github.com/princeton-nlp/WebShop/tree/64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd) | MIT ([notice](eval_data/webshop/LICENSE)) |

Knowledge Weaver's own code is released under the root [LICENSE](LICENSE) (MIT). In `AReaL/`, files we wrote carry an MIT header, upstream files we modified carry a notice, and all other files are unchanged upstream code. ALFWorld and WebShop are not vendored: `scripts/setup_env.sh` installs them at the commits above. `eval_data/` contains only their evaluation tasks. Benchmark data and model weights are subject to their providers' terms.
