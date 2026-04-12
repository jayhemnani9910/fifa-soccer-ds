# Changelog

## [0.2.0](https://github.com/jayhemnani9910/fifa-soccer-ds/compare/v0.1.0...v0.2.0) (2026-04-12)


### Features

* add GitHub AI automation workflows ([2c735d2](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/2c735d262a8d6858fd285cff732e41b57e682525))


### Bug Fixes

* add known-first-party config for ruff isort in CI ([b0712df](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/b0712dfb3e0f3468b9a0699997ae3edb35f3afba))
* **ci:** guard torch_geometric imports so pipeline_full imports without it ([8aa80d1](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/8aa80d16b6805fe49ac2fcba62bbc946b0c12b2f))
* defensive improvements across pipeline and utility modules ([717e2b9](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/717e2b9944e6e07aa957c20c66d6e60abce3378f))
* exclude test_pipeline_full from CI smoke tests (import errors) ([94cb2db](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/94cb2dbf81d1ec0a87c38d3701381877aedba779))
* ignore UP038 rule (removed in newer ruff, still enforced in 0.3.0) ([e9be6f9](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/e9be6f92f92d95503d55725e9fd0b80d90c10729))
* match CI ruff version to local (0.15.9) and format all files ([eb0934c](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/eb0934cb51990787e50d16851100a3551e1780e0))
* pin ruff version in CI and skip problematic GPU deps in test job ([1cf8b3b](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/1cf8b3b58bec5f95a8b25b493cfb6b97e85a644a))
* pin ruff==0.3.0 in CI and fix requirements parsing ([49b7f6c](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/49b7f6c1cf16484bf4d198445397ab363b87909e))
* resolve all CI failures — lint errors, test deps, and config ([18c0715](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/18c07150451009638b01dfff7a702b35ddf61ffb))
* run only smoke-compatible test files in CI ([f95c0b3](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/f95c0b37f4df8bafe47b6106b32f868f0e621cc4))


### Documentation

* add GitHub Pages static site with detection gallery ([59f9827](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/59f9827d8a1f5fe102f8e0d31994e3738897e7b8))
* add manual GitHub settings guide ([827f38b](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/827f38b911cf1fb1278b9b79c38367712bc87c71))
* deploy Pages from docs/ (not repo root) + disable Jekyll templating ([b42d34d](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/b42d34dbba4d71853a909b449d636e8f68814b8f))
* **pages:** before/after gallery with raw input frames + output JSON samples ([bcfeac0](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/bcfeac012b6b3d3fdf1afd7d408628ea91c732f2))
* **pages:** replace misleading close-up (FP on crowd) with midfield action frame ([a4f399d](https://github.com/jayhemnani9910/fifa-soccer-ds/commit/a4f399d34bc21d06b07f2de0168074f2252d81d5))
