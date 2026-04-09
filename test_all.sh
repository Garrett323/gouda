#!/usr/bin/env bash

cargo test
maturin develop --release
python -m pytest tests
