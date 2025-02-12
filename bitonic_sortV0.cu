#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>
#include <cuda_runtime.h>

// Uncomment DEBUG to print intermediate results or VERIFY to check correctness
// #define DEBUG
// #define VERIFY

// Function prototypes for debugging and verification
void debug_print(int *local_array, size_t size, int dimension, int distance);
void sequential_sort_verify(int *array, int *sequential_array, size_t size);

// Comparison function for qsort (standard library sorting)
int compareAscending(const void *a, const void *b)
{
    // Cast pointers and dereference to compare integers in ascending order
    return (*(int *)a - *(int *)b);
}

// Inline device function to swap two integers
// __forceinline__ tells the compiler to inline the swap function to reduce overhead
__device__ __forceinline__ void swap(int &a, int &b)
{
    int temp = a;
    a = b;
    b = temp;
}

// Kernel to perform compare-and-swap between two elements if needed
// Each thread identifies its partner with a bitwise XOR using a given distance
__global__ void compareSwapV0(int *data, size_t size, int dimension, int distance)
{
    // Calculate global thread ID based on thread and block indices
    int tid = threadIdx.x + blockIdx.x * blockDim.x; 
    // Compute partner index using XOR; this creates the bitonic pairing pattern
    int partner = tid ^ distance;                    
    // Determine whether this thread should ensure ascending order or descending order.
    // The decision is based on checking whether the bit at 'dimension' is 0.
    bool isAscending = (tid & (1 << dimension)) == 0;

    // Ensure valid indices and that each pair is handled only once (partner > tid)
    if (tid < size && partner > tid)
    {
        // Compare the two elements. If their order does not match the desired order, swap them.
        if ((data[tid] > data[partner]) == isAscending)
        {
            swap(data[tid], data[partner]); // Inline swap to exchange the elements
        }
    }
}

// Function to perform bitonic sort on an array using CUDA kernels
void bitonicSortV0(int *array, size_t size)
{
    // Create CUDA events for measuring execution time on the GPU
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Retrieve properties of device 0 (e.g., maxThreadsPerBlock, maxGridSize, etc.)
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);

    // Allocate device memory for our data array
    int *d_array;
    cudaMalloc(&d_array, size * sizeof(int));

    // Start timing before data transfer and kernel execution
    cudaEventRecord(start);


    /************************************************************************/

    // Transfer host data to device memory
    cudaMemcpy(d_array, array, size * sizeof(int), cudaMemcpyHostToDevice);

    // Determine the number of threads per block:
    // Use all elements if size is smaller than the device maximum, otherwise use max threads.
    int maxThreadsPerBlock = deviceProp.maxThreadsPerBlock; // Typically 1024
    dim3 threadsPerBlock(min(static_cast<size_t>(size), static_cast<size_t>(maxThreadsPerBlock)));

    // Calculate how many blocks are needed to cover the whole array
    int maxGridSize = deviceProp.maxGridSize[0];
    dim3 blocksPerGrid((size + threadsPerBlock.x - 1) / threadsPerBlock.x);
    if (blocksPerGrid.x > maxGridSize)
    {
        printf("Error: Exceeded maximum grid size\n");
        return;
    }

    // Determine the maximum number of dimensions in the sorting hypercube (log2(size))
    int max_hypercube_dimension = log2(size);

    // The bitonic sort algorithm works by performing multiple compare-swap passes
    // The outer loop iterates through the dimensions, defining the overall order
    for (int dimension = 1; dimension <= max_hypercube_dimension; dimension++)
    {
        // The inner loop adjusts the compare distance within the current dimension
        // It starts at 2^(dimension-1) and halves in each iteration (distance >>= 1)
        for (int distance = 1 << (dimension - 1); distance > 0; distance >>= 1)
        {
            // Launch the kernel for the current dimension and distance
            // Each thread compares and possibly swaps a pair of elements
            compareSwapV0<<<blocksPerGrid, threadsPerBlock>>>(d_array, size, dimension, distance);
        }

#ifdef DEBUG
        // If debugging is enabled, copy array back to host and print current state after each dimension
        cudaMemcpy(array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);
        debug_print(array, size, dimension, 0);
#endif
    }

    // Copy the sorted data back from device to host memory
    cudaMemcpy(array, d_array, size * sizeof(int), cudaMemcpyDeviceToHost);

    /************************************************************************/


    // Record the stop event and synchronize to ensure all work is complete for timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // Calculate and print the elapsed time in milliseconds
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f ms\n", milliseconds);

    // Clean up CUDA events and free device memory
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
    bitonicSortV0(array, size);

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