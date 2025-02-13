#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

// Optional debugging and verification flags
// Uncomment DEBUG to print intermediate results
// Uncomment VERIFY to check correctness against sequential sort
// #define DEBUG
// #define VERIFY

// Function prototypes for debugging and verification
void debug_print(int *local_array, size_t size, int dimension, int distance);
void sequential_sort_verify(int *array, int *sequential_array, size_t size);

// Comparison function used by qsort for verification
// Returns difference between two integers for ascending order sort
int compareAscending(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

// GPU device function to swap two integers
// __forceinline__ hint to compiler to inline this function for performance
// __device__ specifies this function runs on the GPU
__device__ __forceinline__ void swap(int &a, int &b)
{
    int temp = a;
    a = b;
    b = temp;
}

// Kernel for merging sorted sequences within a CUDA block
// This kernel handles the merging phase of bitonic sort within block boundaries
__global__ void intraBlockMerge(int *data, size_t size, int dimension, int max_intra_block_distance)
{
    // Calculate global thread ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size)
        return;

    // Determine sort direction based on thread ID and dimension
    bool isAscending = (tid & (1 << dimension)) == 0;

    // Iterate through decreasing distances, performing compare-swap operations
    for (int distance = max_intra_block_distance; distance > 0; distance >>= 1)
    {
        // Find partner thread using XOR operation
        int partner = tid ^ distance;

        // Compare and swap if needed, ensuring partner has higher index
        if (partner > tid && (data[tid] > data[partner]) == isAscending)
        {
            swap(data[tid], data[partner]);
        }

        // Synchronize threads within block before next iteration
        __syncthreads();
    }
}

// Kernel for global compare-swap operations
// Handles comparisons between elements in different blocks
__global__ void compareSwapV0(int *data, size_t size, int dimension, int distance)
{
    // Calculate global thread ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Find partner using XOR with distance
    int partner = tid ^ distance;
    // Determine sort direction
    bool isAscending = (tid & (1 << dimension)) == 0;

    // Only process valid pairs where partner has higher index
    if (tid < size && partner > tid)
    {
        if ((data[tid] > data[partner]) == isAscending)
        {
            swap(data[tid], data[partner]);
        }
    }
}

// Kernel for sorting elements within each block
// Implements the first phase of bitonic sort within block boundaries
__global__ void intraBlockSort(int *data, size_t size, int max_intra_block_dimension)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= size)
        return;

    // Outer loop: iterate through dimensions
    for (int dimension = 1; dimension <= max_intra_block_dimension; dimension++)
    {
        bool isAscending = (tid & (1 << dimension)) == 0;

        // Inner loop: compare-swap with decreasing distances
        for (int distance = 1 << (dimension - 1); distance > 0; distance >>= 1)
        {
            int partner = tid ^ distance;

            if (partner > tid && (data[tid] > data[partner]) == isAscending)
            {
                swap(data[tid], data[partner]);
            }

            __syncthreads();
        }
    }
}

// Main sorting function that orchestrates the GPU-based bitonic sort
void bitonicSortV1(int *array, size_t size)
{
    // Setup CUDA timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Get GPU device properties for configuration
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // Allocate GPU memory for the array
    int *d_array;
    cudaMalloc(&d_array, size * sizeof(int));

    cudaEventRecord(start);


    /************************************************************************/

    // Transfer host data to device memory
    cudaMemcpy(d_array, array, size * sizeof(int), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    dim3 threadsPerBlock(min(static_cast<size_t>(size), static_cast<size_t>(maxThreadsPerBlock)));

    // Calculate grid size ensuring we don't exceed hardware limits
    int maxGridSize = deviceProp.maxGridSize[0];
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    if (blocksPerGrid.x > maxGridSize)
    {
        printf("Error: Exceeded maximum grid size\n");
        return;
    }

    // Calculate dimensions for sorting phases
    int max_hypercube_dimension = log2(size);
    int max_intra_block_dimension = log2(threadsPerBlock.x);
    int max_intra_block_distance = threadsPerBlock.x / 2;

    // Phase 1: Sort within each block
    intraBlockSort<<<blocksPerGrid, threadsPerBlock>>>(d_array, size, max_intra_block_dimension);

    // Phase 2: Merge sorted sequences across blocks
    for (int dimension = max_intra_block_dimension + 1; dimension <= max_hypercube_dimension; dimension++)
    {
        // Handle larger distances with global compare-swap
        for (int distance = 1 << (dimension - 1); distance > max_intra_block_distance; distance >>= 1)
        {
            compareSwapV0<<<blocksPerGrid, threadsPerBlock>>>(d_array, size, dimension, distance);
        }

        // Merge within blocks
        intraBlockMerge<<<blocksPerGrid, threadsPerBlock>>>(d_array, size, dimension, max_intra_block_distance);
    }

    // Copy the sorted data back from device to host memory
    cudaMemcpy(array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);

    /************************************************************************/


    // Record the stop event and synchronize to ensure all work is complete for timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f ms\n", milliseconds);

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_array);
}

int main(int argc, char **argv)
{
    // Verify the command-line arguments (should be exactly 2: program name and q)
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <q>\n", argv[0]);
        return 1;
    }

    // Convert argument to integer: q is the log2 of the number of elements to sort
    int q = atoi(argv[1]);

    // Calculate the total number of elements to sort (2^q)
    size_t size = 1 << q; // Using bit shifting to compute power of 2

    // Allocate host memory for the array to be sorted
    int *array = (int *)malloc(size * sizeof(int));

    // Seed the random number generator using the current time for varied results
    srand(time(NULL));

    // Fill the array with random integers (0 to 999)
    for (int i = 0; i < size; i++)
    {
        array[i] = rand() % 1000; // Random integer between 0 and 999
    }

#ifdef VERIFY
    // Optionally create a copy of the array to verify sorting later
    int *sequential_array = (int *)malloc(size * sizeof(int));
    memcpy(sequential_array, array, size * sizeof(int));
#endif

    // Optional debug print before sorting (if DEBUG is defined)
    debug_print(array, size, 0, 0);

    // Run the bitonic sort on the array
    bitonicSortV1(array, size);

    // Optional debug print after sorting (if DEBUG is defined)
    debug_print(array, size, q, 0);

#ifdef VERIFY
    // Verify that the sorted array matches what a sequential sort produces
    sequential_sort_verify(array, sequential_array, size);
#endif

    // Free the allocated host memory
    free(array);
    return 0;
}

// Debug function to print the array's current state.
// Printing occurs only when DEBUG is defined during compilation.
void debug_print(int *array, size_t size, int dimension, int distance)
{
#ifdef DEBUG
    printf("\nAfter: Dimension %d, Distance %d\n", dimension, distance);
    for (int i = 0; i < size; i++)
    {
        printf("%d ", array[i]);
    }
    printf("\n");
#endif
}

// Function to verify the correctness of the bitonic sort by comparing
// with the result of the C standard library's sequential qsort function.
void sequential_sort_verify(int *array, int *sequential_array, size_t size)
{
    // Sort using qsort on a copy of the array. This is our reference.
    qsort(sequential_array, size, sizeof(int), compareAscending);

    // Compare each element of the two arrays. Report a mismatch if found.
    bool is_sorted = true;
    for (int i = 0; i < size; i++)
    {
        if (array[i] != sequential_array[i])
        {
            printf("Error: Mismatch at index %d: %d != %d\n", i, array[i], sequential_array[i]);
            is_sorted = false;
            break;
        }
    }

    // Free the verification array and output the final result.
    free(sequential_array);
    printf("\n%s sorting %zu elements\n\n\n", is_sorted ? "SUCCESSFUL" : "FAILED", size);
}