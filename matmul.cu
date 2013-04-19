#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <sys/resource.h>

#define N 10
__global__ void MatMul(float d_A[N][N], float d_B[N][N], float d_C[N][N])
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;   
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (i < N && j < N)
  {
    for (int l = 0; l < N; l++)
    {
      d_C[i][j] = d_C[i][j] + d_A[j][l] * d_B[l][i];
    
    }
  }
}

__global__ void setElement(float d_A[N][N], float d_B[N][N], float d_C[N][N])
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < N && j < N){
  	d_A[i][j] = i * (float)3.2 + j * (float)2.21;
  	d_B[i][j] = i * (float)1.3 + j * (float)3.1;
    d_C[i][j] = (float)0;
  }
}

int main()
{

  int m,n,k;
  m = n = k = N;

  int i,j;
  int ARRAY_SIZE = N * N;

  int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);


  float h_A[N][N], h_B[N][N], h_C[N][N];
  float (*d_A)[N], (*d_B)[N], (*d_C)[N];

  cudaMalloc((void**) &d_A, ARRAY_BYTES);
  cudaMalloc((void**) &d_B, ARRAY_BYTES);
  cudaMalloc((void**) &d_C, ARRAY_BYTES);
  
  // Kernel invocation with one block of N * N * 1 threads
  int numBlocks = 5;
  dim3 threadsPerBlock(N, N);
  setElement<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

  cudaMemcpy(h_A, d_A, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_C, d_C, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  
  
    fprintf(stdout, "Here is the matrix A:\n\n");
  for(i=0;i<m;i++) {
    for(j=0;j<k;j++) {
      fprintf(stdout, "%10.2f ",h_A[i][j]);
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

  
  MatMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

  cudaMemcpy(h_C, d_C, ARRAY_BYTES, cudaMemcpyDeviceToHost);


    fprintf(stdout, "Here is the matrix C:\n\n");
  for(i=0;i<m;i++) {
    for(j=0;j<n;j++) {
      fprintf(stdout, "%10.2f",h_C[i][j]);
    }
    fprintf(stdout, "\n");
  }


  // Clean up memory
  cudaFreeHost(h_A);
  //free(h_B);
  cudaFreeHost(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);


}

//nvcc -o test test.cu 