#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <time.h>



#define N 289

__global__ void MatMul(float d_A[N][N], float d_B[N][N], float d_C[N][N])
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

__global__ void setElement(float d_A[N][N], float d_B[N][N], float d_C[N][N])
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < N && j < N){
  	/* d_A[i][j] = i * (float)3.2 + j * (float)2.21;
 	   d_B[i][j] = i * (float)1.3 + j * (float)3.1;
    */
	
	d_A[i][j] = 1.0;
 	d_B[i][j] = 1.0;
    



    d_C[i][j] = (float)0;
  }
}

int main()
{

  cudaError_t res = cudaSuccess;	

  int m,n,k;
  m = n = k = N;

  int i,j;
  int ARRAY_SIZE = N * N;

  int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);


  float h_A[N][N], h_B[N][N], h_C[N][N];
  float (*d_A)[N], (*d_B)[N], (*d_C)[N];

  res=cudaMalloc((void**) &d_A, ARRAY_BYTES);
  if(res!=cudaSuccess ){
		printf("\nCuda error!");
		return -1;
  }
  
  res=cudaMalloc((void**) &d_B, ARRAY_BYTES);
  if( res!=cudaSuccess ){
		printf("\nCuda error!");
		return -1;
  }

  res=cudaMalloc((void**) &d_C, ARRAY_BYTES);
  if( res!=cudaSuccess ){
		printf("\nCuda error!");
		return -1;
  }


  // Kernel invocation with CONVENIENT amount of blocks
	
  int xThreadsPerBlock=32;
  int yThreadsPerBlock=32;
  int xBlocks = (N+(xThreadsPerBlock-1))/xThreadsPerBlock;
  int yBlocks = (N+(yThreadsPerBlock-1))/yThreadsPerBlock;

  dim3 threadsPerBlock(xThreadsPerBlock,yThreadsPerBlock);
  dim3 numBlocks( xBlocks,yBlocks );

  setElement<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

  cudaDeviceSynchronize();

  res=cudaMemcpy(h_A, d_A, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  if( res!=cudaSuccess){
		printf("\nCuda error!");
		return -1;
  }
  res=cudaMemcpy(h_B, d_B, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    if( res!=cudaSuccess){
		printf("\nCuda error!");
		return -1;
  }
  
  res=cudaMemcpy(h_C, d_C, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    if( res!=cudaSuccess){
		printf("\nCuda error!");
		return -1;
  }
  

    fprintf(stdout, "Here is the matrix A:\n\n");
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

  
  MatMul<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

  cudaDeviceSynchronize();

  res=cudaMemcpy(h_C, d_C, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    if( res!=cudaSuccess){
		printf("\nCuda error!");
		return -1;
  }



    fprintf(stdout, "Here is the matrix C:\n\n");
  for(i=0;i<m;i++) {
    for(j=0;j<n;j++) {
      fprintf(stdout, "%10.2f",h_C[i][j]);
    }
    fprintf(stdout, "\n");
  }


  // Clean up memory
  cudaFreeHost(h_A);
  cudaFreeHost(h_B);
  cudaFreeHost(h_C);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);


}
