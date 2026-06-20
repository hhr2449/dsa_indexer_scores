# Plan

1. Build a self-contained CUDA/PyTorch local evaluator.
2. Validate the extracted official baseline on boundaries, randomized paged mappings,
   and all ten documented shapes.
3. Record baseline timings with CUDA events.
4. Begin page-tiled kernel candidates only after the baseline artifacts exist.
