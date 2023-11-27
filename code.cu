/*
 * Title: CS6023, GPU Programming, Jan-May 2023, Assignment-1
 * Description: Computation of a matrix C = Kronecker_prod(A, B.T)
 *              where A and B are matrices of dimension (m, n) and
 *              the output is of the dimension (m * n, m * n). 
 * Note: All lines marked in --> should be replaced with code. 
 */

#include <cstdio>        // Added for printf() function 
#include <sys/time.h>    // Added to get time of day
#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
using namespace std;

ofstream outfile; // The handle for printing the output

__global__ void per_row_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....
    //Taking Rows of A and B in thread and evaluating
    int rowA = blockIdx.x;
    int rowB = threadIdx.x;
    //To Map the kronocker multiplication in result using formulae
    for(int colA=0;colA<n;colA++)
    {
        for(int colB=0;colB<n;colB++)
        {
		int a = (rowA*n + colB)*m*n;
		int b = (rowB+m*colA);
		int c = rowA*n + colA;
		int d = rowB*n + colB;
            C[a + b] = A[c] * B[d];
        }
    }
}

__global__ void per_column_AB_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....
    //Taking column of matrix A using grid, depends on the column size and column of B directly from n
    int colA = blockIdx.x * blockDim.x + threadIdx.x;
    int colB = threadIdx.y;
    if(colA < n ) //checking out of bound condition
    { 
      for(int rowA = 0;rowA < m;rowA++)
      {
        for(int rowB = 0; rowB < m;rowB++)
          {
            int a = (rowA*n + colB)*m*n;
		int b = (rowB+m*colA);
		int c = rowA*n + colA;
		int d = rowB*n + colB;
                C[a + b] = A[c] * B[d]; //forumlae to map the multiplication as required
          }
      }
    }
}

__global__ void per_element_kernel(long int *A, long int *B, long int *C,long int m, long int n){
    // --> Complete the kernel ....
	//Calculating thread id  so that it will take id properly...
	int a = threadIdx.x*blockDim.y;
	int b = blockIdx.y*blockDim.y*blockDim.x;
	int c = blockIdx.x*gridDim.y*blockDim.y*blockDim.x;
	int d = threadIdx.y;
	int id = a + b + c + d;
    if(id<((m*n)*(m*n))) //checking for out of bound
    {
	int e = (id/(n*m*n));
	int f = (id/m)%n;
	int g = ((id/(m*n))%n);
        C[id] = A[e*n + f]*B[(id%m)*n + g]; // calculating the kronockers multiplication
	
    }
}

/**
 * Prints any 1D array in the form of a matrix
 **/
void printMatrix(long int *arr, long int rows, long int cols, char* filename){
    outfile.open(filename);
    for(long int i = 0; i < rows; i++){
        for(long int j = 0; j < cols; j++){
            outfile<<arr[i * cols + j]<<" ";
        }
        outfile<<"\n";
    }
    outfile.close();
}

/**
 * Timing functions taken from the matrix multiplication source code
 * rtclock - Returns the time of the day 
 * printtime - Prints the time taken for computation 
 **/
double rtclock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday(&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d", stat);
    return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void printtime(const char *str, double starttime, double endtime){
    printf("%s%3f seconds\n", str, endtime - starttime);
}

int main(int argc,char **argv){
    // Variable declarations
    long int m,n;	
    cin>>m>>n;	

    // Host_arrays 
    long int *h_a,*h_b,*h_c;

    // Device arrays 
    long int *d_a,*d_b,*d_c;
	
    // Allocating space for the host_arrays 
    h_a = (long int *) malloc(m * n * sizeof(long int));
    h_b = (long int *) malloc(m * n * sizeof(long int));	
    h_c = (long int *) malloc(m * m * n * n * sizeof(long int));	

    // Allocating memory for the device arrays 
    // --> Allocate memory for A on device 
      cudaMalloc(&d_a,sizeof(long int) * m * n);
    // --> Allocate memory for B on device 
      cudaMalloc(&d_b,sizeof(long int) * m * n);
    // --> Allocate memory for C on device 
      cudaMalloc(&d_c,sizeof(long int) * m * n * m * n);
    // Read the input matrix A 
    for(long int i = 0; i < m * n; i++) {
        cin>>h_a[i];
    }

    //Read the input matrix B 
    for(long int i = 0; i < m * n; i++) {
        cin>>h_b[i];
    }

    // Transfer the input host arrays to the device 
    // --> Copy A from Host to Device
    cudaMemcpy(d_a,h_a,sizeof(long int) * m * n,cudaMemcpyHostToDevice);
    // --> Copy B from Host to Device 
    cudaMemcpy(d_b,h_b,sizeof(long int) * m * n,cudaMemcpyHostToDevice);
    long int gridDimx, gridDimy;
    
    // Launch the kernels
    /**
     * Kernel 1 - per_row_AB_kernel
     * To be launched with 1D grid, 1D block
     * Each thread should process a complete row of A, B
     **/

    // --> Set the launch configuration
    dim3 grid1(m,1,1);
    dim3 block1(m,1,1); 


    double starttime = rtclock();  

    // --> Launch the kernel 
    per_row_AB_kernel<<<grid1,block1>>>(d_a,d_b,d_c,m,n);
    cudaDeviceSynchronize();                                                           

    double endtime = rtclock(); 
	printtime("GPU Kernel-1 time: ", starttime, endtime);  

    // --> Copy C from Device to Host 
    cudaMemcpy(h_c,d_c,sizeof(long int) * m * n * m * n,cudaMemcpyDeviceToHost);
    

    printMatrix(h_c, m * n, m * n,"kernel1.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(int));

    /**
     * Kernel 2 - per_column_AB_kernel
     * To be launched with 1D grid, 2D block
     * Each thread should process a complete column of  A, B
     **/
    
    // --> Set the launch configuration 
    int gridDimx1 = ceil(float(n)/(10));
    dim3 grid2(gridDimx1,1,1);
    dim3 block2(10,n,1);

    starttime = rtclock(); 

    // --> Launch the kernel
    per_column_AB_kernel<<<grid2,block2>>>(d_a,d_b,d_c,m,n); 

    cudaDeviceSynchronize(); 

    endtime = rtclock(); 
  	printtime("GPU Kernel-2 time: ", starttime, endtime);  

    // --> Copy C from Device to Host
    cudaMemcpy(h_c,d_c,sizeof(long int) * m * n * m * n,cudaMemcpyDeviceToHost);

    printMatrix(h_c, m * n, m * n,"kernel2.txt");
    cudaMemset(d_c, 0, m * n * m * n * sizeof(int));

    /**
     * Kernel 3 - per_element_kernel
     * To be launched with 2D grid, 2D block
     * Each thread should process one element of the output 
     **/
    gridDimx = ceil(float(n * n) / 16);
    gridDimy = ceil(float(m * m) / 64);
    dim3 grid3(gridDimx,gridDimy,1);
    dim3 block3(64,16,1);

    starttime = rtclock();  

    // --> Launch the kernel 
    per_element_kernel<<<grid3,block3>>>(d_a,d_b,d_c,m,n);
    cudaDeviceSynchronize();                                                              

    endtime = rtclock();  
	printtime("GPU Kernel-3 time: ", starttime, endtime);  

    // --> Copy C from Device to Host
    cudaMemcpy(h_c,d_c,sizeof(long int) * m * n * m * n,cudaMemcpyDeviceToHost);

    printMatrix(h_c, m * n, m * n,"kernel3.txt");

    return 0;
}
