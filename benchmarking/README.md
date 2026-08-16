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

For the primary throughput comparison, use the same explicit native thread
count for every method and record the machine load and CPU governor. For
example, to compare methods using eight threads:

```bash
RAYON_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  uv run python benchmarking/run.py
```

Also report a single-thread comparison as a secondary result. It separates
algorithmic/runtime overhead from parallel speedup, but it should not replace
the native multithread result when multithreading is a central contribution.
The default Iris dataset is too small to demonstrate meaningful scaling;
include datasets with thousands of rows in the final benchmark matrix.

## Thread scaling

The strong-scaling benchmark runs each thread count in a fresh process so
Rayon and BLAS pools are initialized correctly. It produces raw/summary CSVs
plus a speedup and parallel-efficiency figure in PDF and PNG formats.

```bash
uv run python benchmarking/scaling.py --threads 1,2,4,8,16
```

The defaults benchmark KNN on a fixed synthetic matrix of 2,000 rows, 40
features, and 20% MCAR missingness. Increase `--rows` for a machine with many
cores, because useful scaling requires enough work per thread. Report both
speedup and absolute runtime; speedup alone can conceal a slow baseline.

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

