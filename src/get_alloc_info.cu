#include <stdio.h>
#include <stdlib.h>

__global__ void emptyKernel()
{
}

#define cudaCheck(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      printf("CUDA error: %s - %s(%d)\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

int main(int argc, char* argv[])
{
	// Initialization
	/*----------------------------------------------------------------------------------------*/
	int device = atoi(argv[1]);
	cudaCheck(cudaSetDevice(device));
	cudaSetDeviceFlags(cudaDeviceMapHost);

	int runtime_version;
	int driver_version;
	cudaDeviceProp properties;
	cudaGetDeviceProperties(&properties, device);
	cudaRuntimeGetVersion(&runtime_version);
	cudaDriverGetVersion(&driver_version);

	emptyKernel<<<1,1>>>();
	cudaDeviceSynchronize();
	
	// Pool size
	/*----------------------------------------------------------------------------------------*/
	int pool_size = 1;
	char *h_data;
	break_01: __attribute__((unused));
	cudaHostAlloc((void**)&h_data, 1, cudaHostAllocMapped);
	cudaFreeHost(h_data);
	
	// Maximum allocations and granularity
	/*----------------------------------------------------------------------------------------*/
	char **h_data_array = (char**) malloc(pool_size * sizeof(char*));
	cudaHostAlloc((void**)&h_data_array[0], 1, cudaHostAllocMapped);
	break_02: __attribute__((unused));
	int granularity = 0, iteration = 0, flag = 0;
	while(!flag && iteration < pool_size)
	{
		iteration++; 
		cudaHostAlloc((void**)&h_data_array[iteration], 1, cudaHostAllocMapped);
	}
	for(int i = 0; i <= iteration; i++)
	{
		cudaFreeHost(h_data_array[i]);
	}
	free(h_data_array);
	
	// Size classes
	/*----------------------------------------------------------------------------------------*/
	char *h_data_inf, *h_data_sup;
	int inf_size = granularity, sup_size = granularity, finished = 1, class_finished = 0;
	break_03: __attribute__((unused));
	cudaHostAlloc((void**)&h_data_inf, inf_size, cudaHostAllocMapped);
	while(!finished)
	{
		sup_size = sup_size + granularity;
		cudaHostAlloc((void**)&h_data_sup, sup_size, cudaHostAllocMapped);
		cudaFreeHost(h_data_sup);
		if(class_finished)
		{
			class_finished = 0;
			cudaFreeHost(h_data_inf);
			inf_size = sup_size;
			cudaHostAlloc((void**)&h_data_inf, inf_size, cudaHostAllocMapped);
		}
	}
	cudaFreeHost(h_data_inf);
	
	// Larger allocations
	/*----------------------------------------------------------------------------------------*/
	break_04: __attribute__((unused));
	cudaHostAlloc((void**)&h_data, pool_size + 1, cudaHostAllocMapped);
	cudaFreeHost(h_data);

	// Allocator policy
	/*----------------------------------------------------------------------------------------*/
	char *chunk_1, *chunk_2, *chunk_3, *chunk_4, *chunk_5, *chunk_6, *chunk_7, *chunk_8, *chunk_9, *chunk_10;
	cudaHostAlloc((void**)&chunk_1, granularity * 2, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_2, granularity, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_3, granularity * 2, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_4, granularity, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_5, granularity, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_6, granularity, cudaHostAllocMapped);
	cudaFreeHost(chunk_1);
	cudaFreeHost(chunk_3);
	cudaFreeHost(chunk_5);
	cudaHostAlloc((void**)&chunk_7, granularity, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_8, granularity, cudaHostAllocMapped);
	break_05: __attribute__((unused));
	cudaFreeHost(chunk_2);
	cudaFreeHost(chunk_4);
	cudaFreeHost(chunk_6);
	cudaFreeHost(chunk_7);
	cudaFreeHost(chunk_8);

	// Coalescing support
	/*----------------------------------------------------------------------------------------*/
	cudaHostAlloc((void**)&chunk_1, granularity, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_2, granularity, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_3, granularity, cudaHostAllocMapped);
	cudaFreeHost(chunk_1);
	cudaFreeHost(chunk_2);
	cudaHostAlloc((void**)&chunk_4, granularity * 2, cudaHostAllocMapped);
	break_06: __attribute__((unused));
	cudaFreeHost(chunk_3);
	cudaFreeHost(chunk_4);

	// Splitting support
	/*----------------------------------------------------------------------------------------*/
	cudaHostAlloc((void**)&chunk_1, granularity * 2, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_2, granularity, cudaHostAllocMapped);
	cudaFreeHost(chunk_1);
	cudaHostAlloc((void**)&chunk_3, granularity, cudaHostAllocMapped);
	break_07: __attribute__((unused));
	cudaFreeHost(chunk_2);
	cudaFreeHost(chunk_3);

	// Expansion policy
	/*----------------------------------------------------------------------------------------*/
	int max_allocations = pool_size / granularity;
	h_data_array = (char**) malloc(max_allocations * sizeof(char*));
	cudaHostAlloc((void**)&h_data_array[0], granularity, cudaHostAllocMapped);
	break_08: __attribute__((unused));
	int index;
	for(index = 1; index < max_allocations; index++)
	{
		cudaHostAlloc((void**)&h_data_array[index], granularity, cudaHostAllocMapped);
	}
	for(index = 0; index < max_allocations; index++)
	{
		cudaFreeHost(h_data_array[index]);
	}
	free(h_data_array);


	// Pool usage
	/*----------------------------------------------------------------------------------------*/
	int quarter = pool_size / 4;
	cudaHostAlloc((void**)&chunk_1, quarter, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_2, quarter, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_3, quarter, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_4, quarter, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_5, quarter, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_6, quarter, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_7, quarter, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_8, quarter, cudaHostAllocMapped);
	cudaHostAlloc((void**)&chunk_9, quarter, cudaHostAllocMapped);
	cudaFreeHost(chunk_1);
	cudaFreeHost(chunk_2);
	cudaFreeHost(chunk_5);
	cudaHostAlloc((void**)&chunk_10, quarter, cudaHostAllocMapped);
	break_09: __attribute__((unused));
	cudaFreeHost(chunk_10);

	// Shrinking support
	/*----------------------------------------------------------------------------------------*/
	flag = 0;
	break_10: __attribute__((unused));
	cudaFreeHost(chunk_6);
	cudaFreeHost(chunk_7);
	cudaFreeHost(chunk_8);
	flag = 1;
	cudaFreeHost(chunk_9);
	flag = 2;
	cudaFreeHost(chunk_3);
	cudaFreeHost(chunk_4);

	// Finalization
	/*----------------------------------------------------------------------------------------*/
	cudaDeviceReset();
	return 0;
}