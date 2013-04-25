#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include "gputimer.h"

#define N 5

__global__ void setElement(float d_A[N], float d_B[N], float k)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  if (i < N){
    d_A[i] = i * (float)3.2 + k * (float)2.21;
    d_B[i] = i * (float)1.3 + k * (float)3.1;
    //d_C[i] = (float)0;
  }
}

__global__ void matmul(float d_A[N], float d_B[N], float d_C)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;

  d_C = 0.0;
  if (i < N)
    d_C = d_C + d_A[i] * d_B[i];
}

int main()
{
  cudaError_t res = cudaSuccess;

  int i, j, jj;
  int m, n, k;
  m = n = k = N;

  int ARRAY_SIZE = N;

  int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

  float h_A[N][N], h_B[N][N], h_C[N][N];
  float temp_A[N], temp_B[N], temp_C;
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

  res=cudaMalloc((void**) &d_C, sizeof(float));
  if(res!=cudaSuccess)
  {
    printf("\ncudaMalloc error!");
    return -1;
  }

  int numBlocks = ceil((float)N / (float)512);

  //set initial elements for A and B
  for (i = 0; i < N; i++)
  {
    setElement<<<numBlocks, 512>>>(d_A, d_B, i);
    res=cudaMemcpy(temp_A, d_A, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    if(res!=cudaSuccess)
    {
      fprintf(stdout, "cudaMemcpy error! %d\n", i);
      return -1;
    }

    res=cudaMemcpy(temp_B, d_B, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    if(res!=cudaSuccess)
    {
      printf("cudaMemcpy error!\n");
      return -1;
    }

    for (j = 0; j < N; j++)
    {
      h_A[j][i] = temp_A[j];
      h_B[j][i] = temp_B[j];
    }
  }
  
  //pass to gpu and multiply A and B
  for (i = 0; i < N; i++)  //for every row of A
  {
    for (j = 0; j < N; j++) //transfer each element in row i
    {
      temp_A[j] = h_A[i][j];
    }
    res=cudaMemcpy(d_A, temp_A, ARRAY_BYTES, cudaMemcpyHostToDevice);
    if(res!=cudaSuccess)
    {
      fprintf(stdout, "cudaMemcpy A error! %d\n", i);
      return -1;
    }

    for (j = 0; j < N; j++) //column of B
    {
      for (jj = 0; jj < N; jj++)  //each value in column j
      {
        temp_B[jj] = h_B[jj][j];

      }
      res=cudaMemcpy(d_B, temp_B, ARRAY_BYTES, cudaMemcpyHostToDevice);
      if(res!=cudaSuccess)
      {
        fprintf(stdout, "cudaMemcpy B error! %d\n", i);
        return -1;
      }

      temp_C = (float)0;

      /*res=cudaMemcpy(d_C, temp_C, sizeof(float), cudaMemcpyHostToDevice);
      if(res!=cudaSuccess)
      {
        fprintf(stdout, "cudaMemcpy C error! %d\n", i);
        return -1;
      }*/

      matmul<<<numBlocks, 512>>>(d_A, d_B, temp_C);

      res=cudaMemcpy(temp_C, d_C, sizeof(float), cudaMemcpyDeviceToHost);
      if(res!=cudaSuccess)
      {
        fprintf(stdout, "cudaMemcpy C error! %d\n", i);
        return -1;
      }

      h_C[i][j] = temp_C;

      //fprintf("%f\n", (float)temp_C);
    }
    

  }


  //matrix output
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

  fprintf(stdout, "Here is the matrix C:\n\n");

  for(i=0;i<k;i++) 
  {
    for(j=0;j<n;j++) 
    {
      fprintf(stdout, "%10.2f",h_C[i][j]);
    }
    fprintf(stdout, "\n");
  }


}