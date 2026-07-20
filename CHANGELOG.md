# Changelog

## [0.2.0](https://github.com/jayhemnani9910/fifa-soccer-ds/compare/v0.1.0...v0.2.0) (2026-07-20)


### Features

* add GitHub AI automation workflows ([2c735d2](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/2c735d262a8d6858fd285cff732e41b57e682525))
* **security:** add SSRF-checked target validation and output-path containment ([3640d70](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/3640d70875d61ad10897bced4c03270adbacdbe1))


### Bug Fixes

* add known-first-party config for ruff isort in CI ([b0712df](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/b0712dfb3e0f3468b9a0699997ae3edb35f3afba))
* **api:** report readiness honestly and require an explicit checkpoint ([0ceb5bd](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/0ceb5bd2c977d739673767d0a52abfa31ba5dff2))
* **ci:** guard torch_geometric imports so pipeline_full imports without it ([8aa80d1](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/8aa80d16b6805fe49ac2fcba62bbc946b0c12b2f))
* **ci:** repair the two failing quality gates and the local aggregate hook ([b2bbc89](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/b2bbc89f21de7d288b67e8b31d1e4de26b0502b4))
* **data:** stop authenticating with Kaggle at import time ([33fd8cc](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/33fd8cce2f4b38c346eee7d7d7aa69b56485f505))
* defensive improvements across pipeline and utility modules ([717e2b9](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/717e2b9944e6e07aa957c20c66d6e60abce3378f))
* **dx:** make every documented make target work from a clean checkout ([4883814](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/4883814a1638d2a22e820c73936e2d7246544799))
* exclude test_pipeline_full from CI smoke tests (import errors) ([94cb2db](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/94cb2dbf81d1ec0a87c38d3701381877aedba779))
* ignore UP038 rule (removed in newer ruff, still enforced in 0.3.0) ([e9be6f9](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/e9be6f92f92d95503d55725e9fd0b80d90c10729))
* match CI ruff version to local (0.15.9) and format all files ([eb0934c](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/eb0934cb51990787e50d16851100a3551e1780e0))
* pin ruff version in CI and skip problematic GPU deps in test job ([1cf8b3b](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/1cf8b3b58bec5f95a8b25b493cfb6b97e85a644a))
* pin ruff==0.3.0 in CI and fix requirements parsing ([49b7f6c](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/49b7f6c1cf16484bf4d198445397ab363b87909e))
* resolve all CI failures — lint errors, test deps, and config ([18c0715](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/18c07150451009638b01dfff7a702b35ddf61ffb))
* run only smoke-compatible test files in CI ([f95c0b3](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/f95c0b37f4df8bafe47b6106b32f868f0e621cc4))
* **src:** remove false-success paths and harden artifact handling ([377026e](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/377026e2e923e2973158542b5a159dcaa51f87a2))


### Documentation

* add GitHub Pages static site with detection gallery ([59f9827](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/59f9827d8a1f5fe102f8e0d31994e3738897e7b8))
* add manual GitHub settings guide ([827f38b](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/827f38b911cf1fb1278b9b79c38367712bc87c71))
* add v2.0 session progress log ([effb215](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/effb215bb744e9ae5269a2076db1b5d6441d5fcc))
* deploy Pages from docs/ (not repo root) + disable Jekyll templating ([b42d34d](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/b42d34dbba4d71853a909b449d636e8f68814b8f))
* **pages:** before/after gallery with raw input frames + output JSON samples ([bcfeac0](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/bcfeac012b6b3d3fdf1afd7d408628ea91c732f2))
* **pages:** replace misleading close-up (FP on crowd) with midfield action frame ([a4f399d](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/a4f399d34bc21d06b07f2de0168074f2252d81d5))
* reconcile claims with implementation and record audit evidence ([4dec0f1](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/4dec0f1480b2bf308302cd015471a5ca3be2817b))
* record Phase 1.1 and 1.2 outcome and follow-ups ([f263c5d](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/f263c5df6c61971393dd0527bcf962a88835f740))
* record Phase 1.6 outcome and the clean-environment sweep ([1503faa](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/1503faac30ae05588fabf954af6d807c88e30c89))
