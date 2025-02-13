#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

// #define DEBUG // Uncomment to enable debug print statements
// #define VERIFY // Uncomment to enable verification against sequential sort

// Function prototypes for debugging and verification
void debug_print(int *local_array, size_t size, int dimension, int distance);
void sequential_sort_verify(int *array, int *sequential_array, size_t size);

// Comparison function for qsort (stdlib sorting)
// Used for verifying the correctness of the bitonic sort
int compareAscending(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

// Inline device function to swap two integers
// __forceinline__ suggests the compiler to inline this function for performance
// __device__ specifies that this function runs on the GPU
__device__ __forceinline__ void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

// Kernel for merging sorted sequences within a CUDA block using shared memory
// This kernel performs the merging step of the bitonic sort algorithm within a single block
__global__ void intraBlockMergeShared(int *data, size_t size, int dimension, int max_intra_block_distance) {
    // Declare shared memory
    extern __shared__ int sharedData[];  // Dynamically allocated shared memory

    // Calculate global thread ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Calculate local thread ID within the block
    int local_tid = threadIdx.x;
    // Check if the thread ID is within the bounds of the array size
    if (tid >= size) return;

    // Load data from global memory to shared memory
    // Each thread loads one element from global memory into shared memory
    // If tid is within the array size, load data[tid], otherwise load INT_MAX
    sharedData[local_tid] = (tid < size) ? data[tid] : INT_MAX;
    // Synchronize threads within the block to ensure all data is loaded
    __syncthreads();

    // Determine the sorting order based on the bit at position 'dimension' in the thread ID
    bool isAscending = (tid & (1 << dimension)) == 0;

    // Perform bitonic merge within the block
    for (int distance = max_intra_block_distance; distance > 0; distance >>= 1) {
        // Calculate the partner thread ID within the block
        int partner = local_tid ^ distance;  // Use local_tid for partner calculation

        // Compare and swap elements if needed
        // Only threads with partner > local_tid perform the swap to avoid duplicate swaps
        if (partner > local_tid) {
            // If the elements are not in the correct order, swap them
            if ((sharedData[local_tid] > sharedData[partner]) == isAscending) {
                swap(sharedData[local_tid], sharedData[partner]);
            }
        }
        // Synchronize threads within block before next iteration
        __syncthreads();
    }

    // Write sorted data back to global memory
    // Each thread writes its element from shared memory back to global memory
    if (tid < size) {
        data[tid] = sharedData[local_tid];
    }
}

// Kernel for global compare-swap operations
// Handles comparisons between elements in different blocks
__global__ void compareSwapV0(int *data, size_t size, int dimension, int distance)
{
    // Calculate global thread ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x; // Global thread ID
    // Calculate partner thread ID using XOR operation
    int partner = tid ^ distance;                    // XOR operation to find partner (bitwise complement)
    // Determine the sorting order based on the bit at position 'dimension' in the thread ID
    bool isAscending = (tid & (1 << dimension)) == 0;

    // Only process valid pairs where partner has higher index
    if (tid < size && partner > tid)
    {
        // Compare and swap elements if they are not in the correct order
        if ((data[tid] > data[partner]) == isAscending)
        {
            swap(data[tid], data[partner]); // Use inline swap
        }
    }
}

// Kernel for sorting elements within each block using shared memory
// Implements the first phase of bitonic sort within block boundaries
__global__ void intraBlockSortShared(int *data, size_t size, int max_intra_block_dimension) {
    // Declare shared memory
    extern __shared__ int sharedData[];  // Dynamically allocated shared memory

    // Calculate global thread ID
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    // Calculate local thread ID within the block
    int local_tid = threadIdx.x;
    // Check if the thread ID is within the bounds of the array size
    if (tid >= size) return;

    // Load data from global memory to shared memory
    // Each thread loads one element from global memory into shared memory
    // If tid is within the array size, load data[tid], otherwise load INT_MAX
    sharedData[local_tid] = (tid < size) ? data[tid] : INT_MAX;
    // Synchronize threads within the block to ensure all data is loaded
    __syncthreads();

    // Perform bitonic sort within the block
    for(int dimension = 1; dimension <= max_intra_block_dimension; dimension++) {
        // Determine the sorting order based on the bit at position 'dimension' in the thread ID
        bool isAscending = (tid & (1 << dimension)) == 0;   
             
        // Perform bitonic merge within the block
        for (int distance = 1 << (dimension-1); distance > 0; distance >>= 1) {
            // Calculate the partner thread ID within the block
            int partner = local_tid ^ distance;  // Use local_tid for partner calculation

            // Compare and swap elements if needed
            // Only threads with partner > local_tid perform the swap to avoid duplicate swaps
            if (partner > local_tid) {
                // If the elements are not in the correct order, swap them
                if ((sharedData[local_tid] > sharedData[partner]) == isAscending) {
                    swap(sharedData[local_tid], sharedData[partner]);
                }
            }
            // Synchronize threads within block before next iteration
            __syncthreads();
        }
    }

    // Write sorted data back to global memory
    // Each thread writes its element from shared memory back to global memory
    if (tid < size) {
        data[tid] = sharedData[local_tid];
    }
}

// Main function to perform bitonic sort on the GPU
void bitonicSortV2(int *array, size_t size) {
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Get device properties
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // Allocate device memory
    int *devArray;
    cudaMalloc(&devArray, size * sizeof(int));

    cudaEventRecord(start);  // Start timing

    /************************************************************************/  

    // Copy data from host to device
    cudaMemcpy(devArray, array, size * sizeof(int), cudaMemcpyHostToDevice);

    // Configure kernel launch parameters
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock; // Typically 1024
    dim3 threadsPerBlock(min(static_cast<size_t>(size), static_cast<size_t>(maxThreadsPerBlock)));

    // Calculate grid size
    int maxGridSize = deviceProp.maxGridSize[0]; // Typically 65535
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    // Check if the grid size exceeds the maximum allowed size
    if (blocksPerGrid.x > maxGridSize) {
        printf("Error: Exceeded maximum grid size\n");
        return;
    }

    // Calculate dimensions for sorting phases
    int max_hypercube_dimension = log2(size);
    int max_intra_block_dimension = log2(threadsPerBlock.x);
    int max_intra_block_distance = threadsPerBlock.x / 2; // Half the block size

    // Launch the kernel to sort elements within each block using shared memory
    // The third parameter specifies the amount of shared memory to allocate for each block
    intraBlockSortShared<<<blocksPerGrid, threadsPerBlock, threadsPerBlock.x * sizeof(int)>>>(devArray, size, max_intra_block_dimension);

    // Perform bitonic merge across blocks
    for(int dimension = max_intra_block_dimension + 1; dimension <= max_hypercube_dimension; dimension++) {
        // Perform compare-and-swap operations for distances greater than max_intra_block_distance
        for(int distance = 1 << (dimension-1); distance > max_intra_block_distance; distance >>= 1) 
        {
            compareSwapV0<<<blocksPerGrid, threadsPerBlock>>>(devArray, size, dimension, distance);
        }

        // Merge sorted sequences within each block using shared memory
        intraBlockMergeShared<<<blocksPerGrid, threadsPerBlock, threadsPerBlock.x * sizeof(int)>>>(devArray, size, dimension, max_intra_block_distance);
    }

    // Copy sorted data from device to host
    cudaMemcpy(array, devArray, size * sizeof(int), cudaMemcpyDeviceToHost);

    /************************************************************************/

    // Stop timing after copying data back to CPU
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f ms\n", milliseconds);

    // Free device memory
    cudaFree(devArray);
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

// Main function
int main(int argc, char **argv)
{
    // Verify for correct number of arguments
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s <q>\n", argv[0]);
        return 1;
    }

    // Parse command-line arguments
    int q = atoi(argv[1]); // log2 of local array size

    // Calculate local array size
    size_t size = 1 << q; // 2^q elements per process

    // Allocate memory for local array
    int *array = (int *)malloc(size * sizeof(int));

    // Seed random number generator
    srand(time(NULL)); // Seed based on current time

    // Generate random integers for array
    for (int i = 0; i < size; i++)
    {
        array[i] = rand() % 1000; // Random integers between 0 and 999
    }

    #ifdef VERIFY
    // Create a copy of the array for verification
    int *sequential_array = (int *)malloc(size * sizeof(int));
    memcpy(sequential_array, array, size * sizeof(int));
    #endif

    // Print local array before sorting (if DEBUG is defined)
    debug_print(array, size, 0, 0);

    // Perform bitonic sort
    bitonicSortV2(array, size);

    // Print local array after sorting (if DEBUG is defined)
    debug_print(array, size, q, 0);

    #ifdef VERIFY
    // Verify that the sorted array matches what a sequential sort produces
    sequential_sort_verify(array, sequential_array, size);
    #endif

    // Clean up
    free(array);
    return 0;
}

// Debug function to print the array's current state.
// Printing occurs only when DEBUG is defined during compilation.
void debug_print(int *array, size_t size, int dimension, int distance)
{
    #ifdef DEBUG
    printf("\nAfter: Dimenison %d, Distance %d\n", dimension, distance);
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

    // Compare sorted results
    int is_sorted = 1;
    for (int i = 0; i < size; i++)
    {
        if (array[i] != sequential_array[i])
        {
            printf("Error: Mismatch at index %d: %d != %d\n", i, array[i], sequential_array[i]);
            is_sorted = 0;
            break;
        }
    }

    // Free the verification array and output the final result.
    free(sequential_array);
    printf("\n%s sorting %zu elements\n\n\n", is_sorted ? "SUCCESSFUL" : "FAILED", size);
}