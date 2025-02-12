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

int main(int argc, char **argv) {
    // Verify command line arguments
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <q>\n", argv[0]);
        return 1;
    }

    // Parse input size (q is log2 of array size)
    int q = atoi(argv[1]);         // Convert string to integer
    size_t size = 1 << q;          // Calculate array size as 2^q

    // Allocate and initialize input array
    int *array = (int *)malloc(size * sizeof(int));
    srand(time(NULL));  // Seed random number generator

    // Fill array with random values
    for (int i = 0; i < size; i++) {
        array[i] = rand() % 1000;  // Random values between 0 and 999
    }

    #ifdef VERIFY
        // Create a copy for verification if VERIFY is defined
        int *sequential_array = (int *)malloc(size * sizeof(int));
        memcpy(sequential_array, array, size * sizeof(int));
    #endif

    // Perform radix sort using CUB
    cubRadixSort(array, size);

    #ifdef VERIFY
        // Verify sort correctness if VERIFY is defined
        sequential_sort_verify(array, sequential_array, size);
    #endif

    // Cleanup
    free(array);
    return 0;
}

// Verification function to compare GPU sort against CPU sort
void sequential_sort_verify(int *array, int *sequential_array, size_t size)
{
    // Sort the verification array using standard library qsort
    qsort(sequential_array, size, sizeof(int), compareAscending);

    // Compare the two sorted arrays element by element
    int is_sorted = 1;
    for (int i = 0; i < size; i++)
    {
        if (array[i] != sequential_array[i])
        {
            printf("Error: Mismatch at index %d: %d != %d\n", 
                   i, array[i], sequential_array[i]);
            is_sorted = 0;
            break;
        }
    }

    // Free verification array and print result
    free(sequential_array);
    printf("\n%s sorting %zu elements\n\n\n", 
           is_sorted ? "SUCCESSFUL" : "FAILED", size);
}