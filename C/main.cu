#include <stdio.h>
#include <stdlib.h>
#include "JControlFlow.cu"
#include <time.h>

void initialize(int num_arrays, int array_stride, int * bool_array){
        int tot = num_arrays * array_stride;
        int r;
        for(int i = 0; i < tot; i++) {
                r = rand() % 50;
                if(r < 25)
                        bool_array[i] = 0;
                else
                        bool_array[i] = 1;
        }
}

int isEqual(int num, int * array1, int * array2){
        for(int i=0; i < num; i++) {
                if(array1[i] != array2[i])
                        return i;
        }
        return -1;

}
void host_Control_Flow(int n_iterations, int num_arrays, int array_stride, int * boolean_array)
/*
   Periodic BC, loop through each array, if 1 then flip the parity of idx to the right, else flip parity of idx to the left.
 */
{
        for (int i = 0; i < n_iterations; i++) {
                for (int id = 0; id < num_arrays; id++) {
                        unsigned int array_address = id * array_stride;

                        int curr_val;
                        int side_val;

                        for (int j = 0; j < array_stride; j++) {
                                curr_val = boolean_array[array_address + j];

                                if(curr_val == 0) {
                                        if(j > 0) {
                                                side_val = boolean_array[array_address + j - 1];
                                                boolean_array[array_address + j -1] = (side_val +1) % 2;
                                        }
                                        else{
                                                side_val = boolean_array[array_address + array_stride - 1];
                                                boolean_array[array_address + array_stride - 1] = (side_val +1) % 2;
                                        }
                                }

                                else{
                                        if(j < array_stride -1) {
                                                side_val = boolean_array[array_address + j + 1];
                                                boolean_array[array_address + j + 1] = (side_val +1) % 2;
                                        }
                                        else{
                                                side_val = boolean_array[array_address];
                                                boolean_array[array_address] = (side_val +1) % 2;
                                        }
                                }
                        }
                }
        }
}

int main(int argc, char * argv[]){
        int array_stride = 32;
        int num_arrays   = atoi(argv[1]);
        int num_iterations = 10000;


        cudaDeviceReset();

        cudaEvent_t start, intermediate, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&intermediate);
        cudaEventCreate(&stop);
        float timeForCopy, timeForComputation;

        clock_t cpuStart, cpuStop;
        double CPU_time;

        int status;

        int * h_bool_array;
        int * d_bool_array;

        int * h_device_output;

        int tpb = 256;
        dim3 threads = dim3 (tpb, 1, 1);
        dim3 blocks  = dim3 ((num_arrays + tpb -1)/tpb, 1, 1);

        status = cudaMallocHost(&h_bool_array, sizeof(int) * array_stride * num_arrays);
        status = cudaMalloc(&d_bool_array, sizeof(int) * array_stride * num_arrays);
        status = cudaMallocHost(&h_device_output, sizeof(int) * array_stride * num_arrays);

        initialize(num_arrays, array_stride, h_bool_array);

        cudaDeviceSynchronize();

        cudaEventRecord(start, 0);

        status = cudaMemcpy(d_bool_array, h_bool_array, sizeof(int) * array_stride * num_arrays,cudaMemcpyHostToDevice);

        cudaEventRecord(intermediate, 0); // make data point after copy

        Control_Flow <<< blocks, threads >>> (num_iterations,num_arrays, array_stride, d_bool_array);

        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&timeForCopy, start, intermediate); // time for copy in milliseconds (see http://developer.download.nvidia.com/compute/cuda/4_2/rel/toolkit/docs/online/group__CUDART__EVENT_g14c387cc57ce2e328f6669854e6020a5.html)
        cudaEventElapsedTime(&timeForComputation, intermediate, stop); // time for computation in milliseconds

        status = cudaMemcpy(h_device_output, d_bool_array, sizeof(int) * array_stride * num_arrays,cudaMemcpyDeviceToHost);

        printf("time for GPU copy %f \n",timeForCopy/1000);
        printf("time for GPU compute %f \n",timeForComputation/1000);
        printf("total GPU time %f \n", (timeForCopy + timeForComputation)/1000);

        cpuStart = clock();
        host_Control_Flow(num_iterations, num_arrays, array_stride, h_bool_array);
        cpuStop = clock();
        CPU_time = ((double)(cpuStop - cpuStart))/CLOCKS_PER_SEC;

        printf("time for CPU compute %f \n", CPU_time);

        int test = isEqual(num_arrays * array_stride, h_bool_array, h_device_output);
        if(test == -1) {
                printf("Both CPU and GPU produced the same output \n");
        }
        else{
                printf("CPU ad GPU outputs differ, first instance at idx %d ", test);
        }


        status = cudaMemcpy(h_output, d_output, num_elements * sizeof(int),cudaMemcpyDeviceToHost);

        cudaFree(h_bool_array);
        cudaFree(d_bool_array);
        cudaEventDestroy(start);
        cudaEventDestroy(intermediate);
        cudaEventDestroy(stop);

}
