extern "C"


__global__ void Control_Flow(int n_iterations, int num_arrays, int array_stride, int * boolean_array)
/*
   Periodic BC, loop through each array, if 1 then flip the parity of idx to the right, else flip parity of idx to the left.
 */
{
        int shared_boolean_array[32];

        const unsigned int id = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int array_address = id * array_stride;
        const unsigned int shared_address = 0;

        if(id < num_arrays) {

                int curr_val;
                int side_val;

                for(int j = 0; j < array_stride; j++ ) {
                        shared_boolean_array[shared_address + j] = boolean_array[array_address + j];
                }

                for (int i = 0; i < n_iterations; i++) {
                        for (int j = 0; j < array_stride; j++) {
                                curr_val = shared_boolean_array[shared_address + j];
                                if(curr_val == 0) {
                                        if(j > 0) {
                                                side_val = shared_boolean_array[shared_address + j - 1];
                                                shared_boolean_array[shared_address + j -1] = (side_val +1) % 2;
                                        }
                                        else{
                                                side_val = shared_boolean_array[shared_address + array_stride - 1];
                                                shared_boolean_array[shared_address + array_stride - 1] = (side_val +1) % 2;
                                        }
                                }
                                else{
                                        if(j < array_stride -1) {
                                                side_val = shared_boolean_array[shared_address + j + 1];
                                                shared_boolean_array[shared_address + j + 1] = (side_val + 1) % 2;
                                        }
                                        else{
                                                side_val = shared_boolean_array[shared_address];
                                                shared_boolean_array[shared_address] = (side_val + 1) % 2;
                                        }
                                }
                        }
                }

                for(int j = 0; j < array_stride; j++ ) {
                        boolean_array[array_address + j] = shared_boolean_array[shared_address + j];
                }
        }
}
