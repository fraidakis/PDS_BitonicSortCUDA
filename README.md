# Parallel and Distributed Systems - Exercise 3: CUDA Bitonic Sort

This repository contains multiple CUDA implementations of Bitonic Sort with different optimizations, plus a Radix Sort implementation for performance comparison. The project demonstrates GPU-accelerated sorting strategies and their optimization trade-offs.

## Table of Contents
1. [Introduction](#introduction)
2. [Setup and Requirements](#setup-and-requirements)
3. [Compilation and Usage](#compilation-and-usage)
4. [Directory Structure](#directory-structure)
5. [Algorithm Versions](#algorithm-versions)
6. [Performance Metrics](#performance-metrics)
7. [Verification](#verification)

## Introduction

This project implements three optimized versions of **Bitonic Sort** and a **Radix Sort** on NVIDIA GPUs using CUDA:

- **Bitonic Sort V0**: Baseline implementation with global memory operations
- **Bitonic Sort V1**: Optimized with shared memory for intra-block communication
- **Bitonic Sort V2**: Enhanced with coalesced memory accesses and minimal bank conflicts
- **Radix Sort**: Reference implementation using CUDA Thrust library

## Setup and Requirements

- **NVIDIA GPU** with Compute Capability ≥ 3.5
- **CUDA Toolkit** ≥ 11.0
- **Linux/macOS** environment
- **Make** build system

## Compilation and Usage

### Compilation

Use the provided `Makefile` to build all implementations:
```bash
make all
```

