#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include "gputimer.h"

#define N 5

__global__ void setElement(float d_A[N], float d_B[N], float d_C[N], float k)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i < N){
    d_A[i] = i * (float)3.2 + k * (float)2.21;
    d_B[i] = i * (float)1.3 + k * (float)3.1;
    d_C[i] = (float)0;
  }
}

int main()
{
  cudaError_t res = cudaSuccess;

  int i, j;
  int m, n, k;
  m = n = k = N;

  int ARRAY_SIZE = N;

  int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

  float h_A[N][N], h_B[N][N], h_C[N][N];
  float temp_A[N], temp_B[N], temp_C[N];
  float *d_A, *d_B, *d_C;

  res=cudaMalloc((void**) &d_A, ARRAY_BYTES);
  if(res!=cudaSuccess)
  {
    printf("\ncudaMalloc error!");
    return -1;
  }

  res=cudaMalloc((void**) &d_B, ARRAY_BYTES);
  if(res!=cudaSuccess)
  {
    printf("\ncudaMalloc error!");
    return -1;
  }

  res=cudaMalloc((void**) &d_C, ARRAY_BYTES);
  if(res!=cudaSuccess)
  {
    printf("\ncudaMalloc error!");
    return -1;
  }

  int numBlocks = ceil((float)N / (float)512);

  for (i = 0; i < N; i++)
  {
    setElement<<<numBlocks, 512>>>(d_A, d_B, d_C, i);
    res=cudaMemcpy(temp_A, d_A, sizeof(float), cudaMemcpyDeviceToHost);
    if(res!=cudaSuccess)
    {
      printf("\ncudaMemcpy error!");
      return -1;
    }

    res=cudaMemcpy(temp_B, d_B, sizeof(float), cudaMemcpyDeviceToHost);
    if(res!=cudaSuccess)
    {
      printf("\ncudaMemcpy error!");
      return -1;
    }

    for (j = 0; j < N; j++)
    {
      h_A[j][i] = temp_A[j];
      h_B[j][i] = temp_B[j];
    }
  }
  
  fprintf(stdout, "Here is the matrix A:\n\n");
    for(i=0;i<m;i++) 
    {
      for(j=0;j<k;j++) 
      {
        fprintf(stdout, "%10.2f",h_A[i][j]);
      }
      fprintf(stdout, "\n");
    }
  
  fprintf(stdout, "Here is the matrix B:\n\n");

  for(i=0;i<k;i++) 
  {
    for(j=0;j<n;j++) 
    {
      fprintf(stdout, "%10.2f",h_B[i][j]);
    }
    fprintf(stdout, "\n");
  }


}