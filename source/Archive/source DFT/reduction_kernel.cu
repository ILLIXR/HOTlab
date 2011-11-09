#include "GenerateHologramCUDA_PCI.h"

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel 
// 6, we observe the maximum specified number of blocks, because each thread in 
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
        if (n == 1) 
            threads = 1;
        else
            threads = (n < maxThreads*2) ? n / 2 : maxThreads;
        blocks = n / (threads * 2);
        blocks = min(maxBlocks, blocks);
}



void Reduce(int  n, int maxThreads, int maxBlocks, float* d_idata, float* d_odata, int offset)
{
    cudaThreadSynchronize();
    
    int numBlocks = 0;
    int numThreads = 0;
    getNumBlocksAndThreads(n, maxBlocks, maxThreads, numBlocks, numThreads);

    // execute the kernel
    reduce(n, numThreads, numBlocks, d_idata, d_odata, offset);

    // sum partial block sums on GPU
    int s=numBlocks;
    while(s > 1) 
    {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);

        reduce(s, threads, blocks, d_odata, d_odata, offset);
        s = s / (threads*2);    
    }
    cudaThreadSynchronize();
  
    return;
}




/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)
*/

template <unsigned int blockSize>
__global__ void reduce6(float *g_idata, float *g_odata, unsigned int n, int offset)
{
	extern __shared__ float sdata[];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*(blockSize*2) + tid;
	unsigned int gridSize = blockSize*2*gridDim.x;
	sdata[tid] = 0;
	
	while (i < n) { sdata[tid] += g_idata[i + offset] + g_idata[i+blockSize + offset]; i += gridSize; }
	__syncthreads();
	
	if (blockSize >= 512) { if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads(); }
	if (blockSize >= 256) { if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads(); }
	if (blockSize >= 128) { if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads(); }
	
	if (tid < 32) {
		if (blockSize >= 64) sdata[tid] += sdata[tid + 32];
		if (blockSize >= 32) sdata[tid] += sdata[tid + 16];
		if (blockSize >= 16) sdata[tid] += sdata[tid + 8];
		if (blockSize >= 8) sdata[tid] += sdata[tid + 4];
		if (blockSize >= 4) sdata[tid] += sdata[tid + 2];
		if (blockSize >= 2) sdata[tid] += sdata[tid + 1];
	}
	if (tid == 0) g_odata[blockIdx.x + offset] = sdata[0];
}


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////

void reduce(int size, int threads, int blocks, float *d_idata, float *d_odata, int offset)
{
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);
    int smemSize = threads * sizeof(float);

        switch (threads)
        {
        case 512:
            reduce6<512><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, offset); break;
        case 256:
            reduce6<256><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, offset); break;
        case 128:
            reduce6<128><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, offset); break;
        case 64:
            reduce6<64><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, offset); break;
        case 32:
            reduce6<32><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, offset); break;
        case 16:
            reduce6<16><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, offset); break;
        case  8:
            reduce6<8><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, offset); break;
        case  4:
            reduce6<4><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, offset); break;
        case  2:
            reduce6<2><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, offset); break;
        case  1:
            reduce6<1><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, size, offset); break;
        }

}