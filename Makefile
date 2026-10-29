# Checks mirror README.md, tests/regression/README.md and .github/workflows/ci.yml.
# Run `make help` for available targets. Default checks do not load ONNX models.
.DEFAULT_GOAL := all

# ONNX pipelines need about 1 GiB each. Keep suites serial even with `make -j`.
.NOTPARALLEL:

# Extra regression runner options, for example:
#   make regression REGRESSION_ARGS='--only td3 docsaid'
#   make regression-all REGRESSION_ARGS='--report target/regression-observed.json'
# Private reports must stay inside files/. Never use --record for routine checks.
REGRESSION_ARGS ?=

.PHONY: help all check fmt clippy check-scripts test test-rust test-python \
	test-unit test-attempts test-cli test-all test-models test-models-rust \
	download-models regression regression-all regression-strict regression-private \
	doc build verify verify-all clean

help: ## List available commands
	@awk 'BEGIN {FS = ":.*## "; print "Usage: make [target]\n"} /^[a-zA-Z0-9_-]+:.*## / {printf "  %-22s %s\n", $$1, $$2}' Makefile
	@printf '\nModel tests require: make download-models (explicit download + SHA-256 verification).\n'
	@printf 'Regressions require Linux, Python 3.10+ and GNU /usr/bin/time.\n'
	@printf 'Use REGRESSION_ARGS for runner options such as --only, --timeout and --report.\n'

all: check test ## Run static checks and all model-free tests (default)

check: fmt clippy check-scripts ## Run formatting, Clippy and shell syntax checks

fmt: ## Check Rust formatting without modifying files
	cargo fmt --all -- --check

clippy: ## Check all Rust targets; treat warnings as errors
	cargo clippy --locked --all-targets -- -D warnings

check-scripts: ## Check shell script syntax without executing scripts
	@for script in scripts/*.sh; do bash -n "$$script" || exit $$?; done

test: test-rust test-python ## Run all model-free Rust tests and Python harness self-tests

test-rust: ## Run Rust unit and integration tests without model inference
	cargo test --locked --all-targets

test-python: ## Run Python regression harness self-tests (standard library only)
	python3 -m unittest discover -s tests/regression -p 'test_*.py' -v

test-unit: ## Run binary-crate tests, including model-free orchestration
	cargo test --locked --bin mrz_scanner

test-attempts: ## Run only model-free attempt orchestration tests
	cargo test --locked --bin mrz_scanner attempt_tests::

test-cli: ## Run only model-free CLI integration tests
	cargo test --locked --test cli

test-all: test test-models ## Run all public tests, including ONNX and slow regressions

# Models must already exist: test targets never download them or silently skip.
# Ignored Rust tests are model tests, not benchmarks; no benchmark suite exists.
test-models: ## Run serial Rust ONNX tests and full public regressions via project script
	./scripts/test_models.sh

test-models-rust: ## Run only ignored Rust ONNX tests serially in release mode
	cargo test --locked --release --all-targets -- --ignored --test-threads=1

download-models: ## Download missing/invalid models and verify pinned SHA-256 hashes
	./scripts/download_models.sh

regression: build ## Run public image regressions without the two slow cases
regression-all: build ## Run all public image regressions, including large/panoramic inputs
regression-strict: build ## Run all public contracts strictly (known OCR XFAILs also fail)
regression-private: build ## Check local private images/baseline in files/ (explicit opt-in)

regression-all: REGRESSION_FLAGS = --include-slow
regression-strict: REGRESSION_FLAGS = --include-slow --strict-contracts
regression-private: REGRESSION_FLAGS = --private

# Resolve Cargo's actual output directory, including CARGO_TARGET_DIR overrides.
# Reports do not replace the baseline; private inputs are never in aggregate targets.
regression regression-all regression-strict regression-private:
	@target_dir="$$(cargo metadata --locked --no-deps --format-version 1 | \
		python3 -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])')" || exit $$?; \
	python3 tests/regression/run.py --binary "$$target_dir/release/mrz_scanner" \
		$(REGRESSION_FLAGS) $(REGRESSION_ARGS)

doc: ## Generate Rust documentation without dependency docs
	cargo doc --locked --no-deps

build: ## Build the release CLI
	cargo build --locked --release

verify: check test doc build ## Run model-free checks/tests, documentation and release build

verify-all: verify test-models ## Full public verification, including ONNX and slow regressions

clean: ## Remove Cargo build artifacts
	cargo clean
