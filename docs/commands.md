# Local evaluator commands

Run all commands from the repository root.

## Build

```bash
./harness/build.sh src/baseline.cu build/libbaseline.so
./harness/build.sh src/kernel.cu build/libkernel.so
```

`harness/build.sh` uses `/usr/local/cuda-12.4/bin/nvcc` when available. Set
`NVCC=/path/to/nvcc` to select another CUDA compiler; it must support `sm_90a`.

## Correctness

```bash
mkdir -p runs
python harness/test.py --lib build/libbaseline.so --all 2>&1 | tee runs/baseline-correctness.txt
python harness/test.py --lib build/libkernel.so --all 2>&1 | tee runs/kernel-correctness.txt
```

Run one of the ten documented testcases with:

```bash
python harness/test.py --lib build/libbaseline.so --case 5
python harness/test.py --lib build/libkernel.so --case 5
```

## Performance

```bash
python harness/benchmark.py --lib build/libbaseline.so --all 2>&1 | tee runs/baseline-benchmark.txt
python harness/benchmark.py --lib build/libkernel.so --all 2>&1 | tee runs/kernel-benchmark.txt
```

Run one documented testcase with:

```bash
python harness/benchmark.py --lib build/libbaseline.so --case 5
python harness/benchmark.py --lib build/libkernel.so --case 5
```

Each benchmark performs 30 warmups and 100 timed iterations using CUDA events.
The command updates `benchmark.csv`, keyed by library name and testcase.
