#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include "gputimer.h"

#define N 800
__global__ void MatMul(float **d_A, float d_B[N][N], float d_C[N][N])
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;   
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (i < N && j < N)
  {
    for (int l = 0; l < N; l++)
    {
      //d_C[i][j] = d_C[i][j] + d_A[j][l] * d_B[l][i];
      d_C[i][j] = d_C[i][j] + d_A[i][l] * d_B[l][j];
    }
  }
}

__global__ void setElement(float d_A[N], float d_B[N], float d_C, int k)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	//int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < N && j < N){
  	d_A[i] = i * (float)3.2 + k * (float)2.21;
  	d_B[i] = k * (float)1.3 + i * (float)3.1;
    d_C[i] = (float)0;
  }
}

int main()
{
  GpuTimer timer;

  //int m,n,k;
  //m = n = k = N;

  int i, j, k;
  int ARRAY_SIZE = N * N;

  int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);


  float h_A[N][N], h_B[N][N], h_C[N][N];
  float *d_A[N], (*d_B)[N], (*d_C)[N];

  cudaMalloc((void**) &d_A, ARRAY_BYTES);
  cudaMalloc((void**) &d_B, ARRAY_BYTES);
  //cudaMalloc((void**) &d_C, ARRAY_BYTES);
  
  // Kernel invocation with least amount of blocks
  //int numBlocks;
  int block_x = ceil((float)N / (float)512);
  int block_y = 1; //single row

  //ceil((float)N / (float)22);

  dim3 numBlocks(block_x, block_y);
  

  //dim3 threadsPerBlock(22, 22);
  for (i = 0; i < N; i++)
  {
    setElement<<<numBlocks, 512>>>(d_A, d_B, d_C, i); //single row of 512 threads
    for (j = 0; j < N; j++)
    {
      cudaMemcpy(h_A[i][j], d_A[j], sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(h_B[i][j], d_B[j], sizeof(float), cudaMemcpyDeviceToHost);
    }
  }

  for (i = 0; i < N; i++)
  {
    for(j = 0; j < N; j++)
    {
      cudaMemcpy(d_A[j], h_A[i][j], sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(d_B[j], h_B[i][j], sizeof(float), cudaMemcpyHostToDevice);
    }

    for (j = 0; j < N; j++)
    {
      MatMul<<<numBlocks, 512>>>(d_A, d_B, d_C, i);
    }

  }
  

    /*timer.Start();
  setElement<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
  MatMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);
  timer.Stop();*/


/*  cudaMemcpy(h_A, d_A, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_C, d_C, ARRAY_BYTES, cudaMemcpyDeviceToHost);*/
  
  
/*    fprintf(stdout, "Here is the matrix A:\n\n");
  for(i=0;i<m;i++) {
    for(j=0;j<k;j++) {
      fprintf(stdout, "%10.2f",h_A[i][j]);
    }
    fprintf(stdout, "\n");
  }
  fprintf(stdout, "Here is the matrix B:\n\n");
  for(i=0;i<k;i++) {
    for(j=0;j<n;j++) {
      fprintf(stdout, "%10.2f",h_B[i][j]);
    }
    fprintf(stdout, "\n");
  }

    fprintf(stdout, "Here is the matrix C:\n\n");
  for(i=0;i<m;i++) {
    for(j=0;j<n;j++) {
      fprintf(stdout, "%10.2f",h_C[i][j]);
    }
    fprintf(stdout, "\n");
  }*/

  printf("Time elapsed = %g ms\n", timer.Elapsed());


  // Clean up memory
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);


}
