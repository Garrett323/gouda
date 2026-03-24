#!/usr/bin/env bash

maturin develop
cargo test
pytest tests
