# Parallel Pitch Tracking Algorithm
NTHU Parallel Programming Final Project.

Use ACF + medianfilter as algorithm.

- Input: wav file
- Output: pitch array, where each element is the semitone of one frame. (Default frame duration: 32 ms)

One can copy the pitch array to `mymidi.cpp`'s pitch and generate corresponding midi file. Note that there might be some noise between the pitch.

## MPI

Use MPI to run multiple threads, each thread is responsible for `num_frames / # of threads` frames.

## CUDA

Use CUDA to run kernel function of calculating ACF of each frame. Basically,
- N = 4, 4 thread computes a frame
- F = 8, 8 frames a block
- T = 32, 32 threads a block

The optimized version take original signal data with share memory.