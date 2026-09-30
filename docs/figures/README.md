# Architecture figures for the midterm report

Add this `figures` folder under the repository's `docs/` directory.

- `ticknet-l-architecture.svg`: use as the architecture figure for **TickNet-L
  v1 / Model 4.2**.  It shows the full backbone and the compressed PDP block.
- `ticknet-model-comparison.svg`: use as the comparison figure for the original
  TickNet-Basic, TickNet-L and TickNet-C.  It distinguishes the two candidates
  and marks clearly that only L has a completed full training run here.

Suggested report captions:

1. **Figure: TickNet-L v1 architecture.** TickNet-L uses seven compressed PDP
   blocks, a 24-channel stem, mixed depthwise 3×3/5×5 kernels in later stages,
   a 768-channel head and a five-class classifier.
2. **Figure: Comparison with the inherited TickNet backbone and candidate C.**
   TickNet-L changes channel allocation, block depth, late-stage depthwise
   kernels and head width; candidate C instead configures the original
   TickNet/FR-PDP implementation with a larger nine-block schedule.
