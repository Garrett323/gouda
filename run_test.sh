#!/usr/bin/env bash
cargo test --release
maturin develop --release
python -m pytest tests
