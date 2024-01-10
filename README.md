# Parallel Pitch Tracking Algorithm
NTHU Parallel Programming Final Project.

Use ACF + medianfilter as algorithm.

## MPI

Use MPI to run multiple threads, each thread is responsible for `num_frames / # of threads` frames.

## CUDA

Use CUDA to run kernel function of calculating ACF of each frame. Basically,
- N = 4, 4 thread computes a frame
- F = 8, 8 frames a block
- T = 32, 32 threads a block

The optimized version take original signal data with share memory.