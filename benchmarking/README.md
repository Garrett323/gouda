# Benchmarking Gouda

This directory contains the reproducible experiment and figure-generation
pipeline used to compare Gouda with reference imputers.

## Methodology

- UCI datasets are fetched by immutable numeric repository ID.
- MCAR uses an independent Bernoulli draw for every originally observed cell.
  MAR and MNAR use `swiss-cheese` and explicitly fail if its installed version
  is incompatible with the active Python interpreter.
- Each algorithm receives the same missingness mechanism, rate, and seed, so
  comparisons are paired.
- If an estimator exposes `random_state` through the sklearn parameter API,
  the missingness seed is also assigned to the estimator unless explicitly
  overridden in `model_params`.
- Timing uses `time.perf_counter_ns`. Garbage collection is disabled only
  around each measured fit/transform pair.
- Every seed has an unmeasured warm-up. Repetitions quantify timing noise;
  seeds are the independent units used for reported variability.
- Numerical accuracy is pooled normalized RMSE: each error is divided by its
  feature's observed range before the RMSE is calculated over artificially
  masked cells. Constant numerical columns are excluded.
- Categorical accuracy is the proportion of falsely classified artificially
  masked cells (PFC).
- Numerical and categorical errors remain separate. They are not added into
  an arbitrary composite score.
- The runner rejects output with the wrong shape or modified observed numeric
  values.

Gouda's experimental `MissForest` is deliberately absent because the current
implementation changes observed values.

## Running experiments

Run commands from any directory; paths are resolved relative to this file by
default.

```bash
uv run python benchmarking/run.py
uv run python benchmarking/run.py -e knn knn-sk
uv run python benchmarking/run.py --config benchmarking/config.yaml --output benchmarking/results
```

For stable timing, pin native thread counts and record the machine load and CPU
governor used for the published run, for example:

```bash
RAYON_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python benchmarking/run.py
```

The output directory contains:

- `raw_results.csv`: one row per experiment, dataset, missing rate, seed, and
  timing repetition;
- `summary.csv`: seed-level accuracy and median-timing summaries, followed by
  mean, median, and sample standard deviation across seeds;
- `metadata.json`: timestamp, platform, Python/package versions, hostname, Git
  commit/dirty state, and relevant thread settings.

Retain all three files with a paper artifact. The raw file is the auditable
source for confidence intervals or alternative statistical analyses.

## Generating figures

```bash
uv run python benchmarking/plotting.py
uv run python benchmarking/plotting.py benchmarking/results/summary.csv \
  --output benchmarking/figures --dataset Iris
```

The plotter produces both 300-DPI PNG and editable PDF files. Each dataset gets
an error/runtime panel with seed standard-deviation bands, and the complete run
gets an accuracy–runtime trade-off plot. The palette is color-vision-friendly,
markers also distinguish methods in grayscale, and PDF text remains editable.

## Publication checklist

1. Choose datasets that cover numerical, categorical, mixed, small, and large
   cases; justify inclusion and report row/feature counts.
2. Run MCAR, MAR, and MNAR as separate configured experiments where the
   scientific claim concerns all three mechanisms.
3. Use at least 10 independent missingness seeds for final confidence
   intervals. Increase timing repetitions only to stabilize runtime estimates.
4. Pin thread counts and use an otherwise idle, fixed-frequency machine.
5. Archive the exact configuration, raw results, metadata, commit hash, and
   built wheel with the journal artifact.
6. Do not compare timing across different machines or thread policies.
7. Report failures and timeouts rather than silently dropping an algorithm.
