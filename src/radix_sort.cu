#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cub/cub.cuh>  // NVIDIA's CUB library for high-performance GPU primitives
#include <time.h>

// Uncomment to enable verification against sequential sort
// #define VERIFY

void sequential_sort_verify(int *array, int *sequential_array, size_t size);

// Comparison function for qsort used in verification
// Returns difference between integers for ascending order sort
int compareAscending(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

// Main radix sort function using NVIDIA's CUB library
void cubRadixSort(int *h_array, size_t size) {
    // Allocate device memory for input and output arrays
    int *d_keys_in;   // Input array on GPU
    int *d_keys_out;  // Output array on GPU
    cudaMalloc(&d_keys_in, size * sizeof(int));
    cudaMalloc(&d_keys_out, size * sizeof(int));

    // Create CUDA events for timing measurement
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start timing
    cudaEventRecord(start);

    // Transfer input array from host to device
    cudaMemcpy(d_keys_in, h_array, size * sizeof(int), cudaMemcpyHostToDevice);

    // CUB sorting preparation
    void *d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    // First call to determine required temporary storage size
    // This is a CUB requirement - we need to query the size first
    cub::DeviceRadixSort::SortKeys(nullptr, temp_storage_bytes, 
                                  d_keys_in, d_keys_out, size);

    // Allocate temporary storage required by CUB
    cudaMalloc(&d_temp_storage, temp_storage_bytes);

    // Perform the actual radix sort
    // This sorts the array in-place on the GPU
    cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, 
                                  d_keys_in, d_keys_out, size);

    // Transfer sorted array back to host
    cudaMemcpy(h_array, d_keys_out, size * sizeof(int), cudaMemcpyDeviceToHost);

    // Stop timing and calculate duration
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("%f ms\n", milliseconds);

    // Cleanup GPU resources
    cudaFree(d_keys_in);
    cudaFree(d_keys_out);
    cudaFree(d_temp_storage);
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

    // Run the bitonic sort on the array
    cubRadixSort(array, size);

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