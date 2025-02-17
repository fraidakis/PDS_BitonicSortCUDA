// Added pinned memory allocation for the input array in main function
// Added __shfl_sync for distances < warpSize in intraBlock kernels


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

/* 
    Function to swap two integer values
    This function is used to swap two integer values
*/
__device__ __forceinline__ void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

/* 
    Function to perform a wrap merge operation within a warp
    This function is used to merge elements within a warp using shuffle operations
*/
__device__ __forceinline__ int intraWrapMerge(int myVal, bool isAscending, int local_tid, int distance) {
    for (; distance > 0; distance >>= 1) {
        int partner_tid = local_tid ^ distance;
        int partner_val = __shfl_sync(0xFFFFFFFF, myVal, partner_tid);
        
        // Substituting the conditional swap with arithmetic operations to avoid branching
        bool pred = (local_tid < partner_tid) == isAscending;
        int min_val = min(myVal, partner_val);
        int max_val = max(myVal, partner_val);
        myVal = pred ? min_val : max_val;
    }
    return myVal;
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
    sharedData[local_tid] = (tid < size) ? data[tid] : INT_MAX;
    __syncthreads();

    // Determine the sorting order based on the bit at position 'dimension' in the thread ID
    bool isAscending = (tid & (1 << dimension)) == 0;

    // First loop: for distances >= warpSize using shared memory and __syncthreads
    for (int distance = max_intra_block_distance; distance >= warpSize; distance >>= 1) {
        int partner = local_tid ^ distance;
        if (partner > local_tid) {
            if ((sharedData[local_tid] > sharedData[partner]) == isAscending) {
                swap(sharedData[local_tid], sharedData[partner]);
            }
        }
        __syncthreads();
    }

    // Write sorted data back to global memory.
    if (tid < size) {
        data[tid] = intraWrapMerge(sharedData[local_tid], isAscending, local_tid, warpSize/2);
    }
}

// Kernel for global compare-swap operations
// Handles comparisons between elements in different blocks
__global__ void compareSwapV0(int *data, size_t size, int dimension, int distance)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x; // Global reduced thread ID

    if (i >= size / 2) return; // We launch only size/2 threads

    // Compute log2(distance) using __ffs (find-first-set) intrinsic.
    // Since distance is a power of 2, __ffs(distance) returns log2(distance) + 1.
    int log_d = __ffs(distance) - 1;

    // Key idea: For each phase, valid comparisons are performed on indices 
    // in the first half of each block of 2*distance elements. 
    // We want to map our reduced thread index (i in [0, size/2)) to an actual array index (tid)
    // that is in the lower half of its 2*distance block.
    //
    // Here’s how the mapping works:

    // 1. Compute the “group” of indices: each group has size = 2*distance.
    //    The group index is given by dividing i by distance.
    int group = i >> log_d;  // equivalent to i / distance.

    // 2. Compute the offset within that group. Since there are exactly 'distance' indices
    //    in the first half of a group, we take i mod distance.
    int offset = i & ((1 << log_d) - 1); // equivalent to i % distance.

    // 3. Compute the full index (tid) in the original array:
    //    Each group starts at index: group * (2 * distance) and then add the offset.
    int tid = group * (2 * distance) + offset;

    // Following remains the same as in compareSwapV0
    int partner = tid ^ distance;

    bool isAscending = (tid & (1 << dimension)) == 0;

    if ((data[tid] > data[partner]) == isAscending)
    {
        swap(data[tid], data[partner]);
    }
}


// Kernel for sorting elements within each block using shared memory
// Implements the first phase of bitonic sort within block boundaries
__global__ void intraBlockSortShared(int *data, size_t size, int max_intra_block_dimension) {
    extern __shared__ int sharedData[];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;
    if (tid >= size) return;

    sharedData[local_tid] = (tid < size) ? data[tid] : INT_MAX;
    __syncthreads();

    for(int dimension = 1; dimension <= max_intra_block_dimension; dimension++) {
        bool isAscending = (tid & (1 << dimension)) == 0;

        int distance;
        for (distance = 1 << (dimension - 1); distance >= warpSize; distance >>= 1) {
            int partner = local_tid ^ distance;
            if (partner > local_tid) {
                if ((sharedData[local_tid] > sharedData[partner]) == isAscending) {
                    swap(sharedData[local_tid], sharedData[partner]);
                }
            }
            __syncthreads();
        }

        sharedData[local_tid] = intraWrapMerge(sharedData[local_tid], isAscending, local_tid, distance);
        __syncthreads();
    }

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
    // Configure kernel launch parameters
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock;
    size_t requiredThreads = size / 2;  // We need size/2 threads for compareSwap

    dim3 threadsPerBlock(min(requiredThreads, static_cast<size_t>(maxThreadsPerBlock)));
    // For compareSwap kernels
    dim3 blocksPerGridCompareSwap(requiredThreads / threadsPerBlock.x);
    // For other kernels
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    
    // Check if the grid size exceeds the maximum allowed size
    if (blocksPerGrid.x > deviceProp.maxGridSize[0]) {
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
            compareSwapV0<<<blocksPerGridCompareSwap, threadsPerBlock>>>(devArray, size, dimension, distance);
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

    // Replace regular malloc with pinned memory allocation
    int *array;
    cudaHostAlloc((void**)&array, size * sizeof(int), cudaHostAllocDefault);


    // Seed the random number generator using the current time for varied results
    srand(time(NULL));

    // Fill the array with random integers (0 to 999)
    for (int i = 0; i < size; i++)
    {
        array[i] = rand() % 1000; // Random integer between 0 and 999
    }

#ifdef VERIFY
    // Use pinned memory for verification array as well
    int *sequential_array;
    cudaHostAlloc((void**)&sequential_array, size * sizeof(int), cudaHostAllocDefault);
    memcpy(sequential_array, array, size * sizeof(int));
#endif

    // Optional debug print before sorting (if DEBUG is defined)
    debug_print(array, size, 0, 0);

    // Run the bitonic sort on the array
    bitonicSortV2(array, size);

    // Optional debug print after sorting (if DEBUG is defined)
    debug_print(array, size, q, 0);

#ifdef VERIFY
    // Verify that the sorted array matches what a sequential sort produces
    sequential_sort_verify(array, sequential_array, size);
#endif

    // Replace free with cudaFreeHost
    cudaFreeHost(array);
#ifdef VERIFY
    cudaFreeHost(sequential_array);
#endif

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

    printf("\n%s sorting %zu elements\n\n\n", is_sorted ? "SUCCESSFUL" : "FAILED", size);
}