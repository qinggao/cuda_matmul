
#include <stdio.h>

#define N 5
__global__ void MatAdd(float d_A[N][N], float d_B[N][N], float d_C[N][N])
{
  int i = threadIdx.x;
  int j = threadIdx.y;
  if (i < N && j < N){
    d_C[i][j] = d_A[i][j] + d_B[i][j];
  }
  
}

__global__ void setElement(float d_A[N][N], float d_B[N][N], float d_C[N][N])
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < N && j < N){
  	d_A[i][j] = i * 3.2 + j * 2.21;
  	d_B[i][j] = i * 1.3 + j * 3.1;
  }
}


int main()
{
  int ARRAY_SIZE = N * N;

  int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);

/*  int h_l,h_m,h_n,h_k;
  int d_l,d_m,d_n,d_k;
*/
/*  h_m = atoi((const char *)argv[1]);
  h_n = atoi((const char *)argv[2]);
  h_k = atoi((const char *)argv[3]);*/

	//float h_C[N][N];

  float h_A[N][N], h_B[N][N], h_C[N][N];
  float (*d_A)[N], (*d_B)[N], (*d_C)[N];

  cudaMalloc(&d_A, ARRAY_BYTES);
  cudaMalloc((void**) &d_B, ARRAY_BYTES);
  cudaMalloc((void**) &d_C, ARRAY_BYTES);
  
  // Kernel invocation with one block of N * N * 1 threads
  int numBlocks = 1;
  dim3 threadsPerBlock(N, N);
  setElement<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

  cudaMemcpy(h_A, d_A, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_B, d_B, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  
  for (int i=0; i < N; i++)
  {
	for (int j=0; j < N; j++)
	{
		printf("%f", h_A[i][j]);
		printf("  %f\n", h_B[i][j]);
	}
  }
  
  
  MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C);

  cudaMemcpy(h_C, d_C, ARRAY_BYTES, cudaMemcpyDeviceToHost);
  for (int i=0; i < N; i++)
  {
  for (int j=0; j < N; j++)
  {
    printf("%f\n", h_C[i][j]);
  }
  }


}

//nvcc -o test test.cu 