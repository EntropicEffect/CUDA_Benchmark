import static jcuda.driver.JCudaDriver.*;
import java.util.Random;
import java.util.concurrent.TimeUnit;
import java.io.*;
import java.util.Scanner;
import jcuda.*;
import jcuda.driver.*;


public class main
{
    public static void main(String args[]) throws IOException

    {
        Scanner array_req = new Scanner(System.in);
        System.out.println("Enter num of arrays: ");
        int num_arrays = array_req.nextInt();
        int array_stride = 32;
        int num_iterations = 10000;

        JCudaDriver.setExceptionsEnabled(true);

        String ptxFileName = preparePtxFile("/home/will/git_repos/CUDA_Benchmark/Java/src/main/java/JControlFlow.cu");
        System.out.println(ptxFileName);



        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);


        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);


        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, "Control_Flow");



        int[] hostBool = new int[num_arrays * array_stride];
        int[] hostDeviceOut = new int[num_arrays * array_stride];

        initialize(num_arrays, array_stride, hostBool);

        CUdeviceptr deviceBoolArray = new CUdeviceptr();
        cuMemAlloc(deviceBoolArray, (num_arrays * array_stride) * Sizeof.INT);


        long startGPUtime = System.nanoTime();
        cuMemcpyHtoD(deviceBoolArray, Pointer.to(hostBool), (num_arrays * array_stride) * Sizeof.INT);


        Pointer kernelParameters = Pointer.to(
                Pointer.to(new int[]{num_iterations}),
                Pointer.to(new int[]{num_arrays}),
                Pointer.to(new int[]{array_stride}),
                Pointer.to(deviceBoolArray)
        );

        int blockSizeX = 256;
        int gridSizeX = (int)Math.ceil((double)(num_arrays * array_stride) / blockSizeX);
        cuLaunchKernel(function,
                gridSizeX,  1, 1,      // Grid dimension
                blockSizeX, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        
        cuCtxSynchronize();

        cuMemcpyDtoH(Pointer.to(hostDeviceOut), deviceBoolArray, num_arrays * array_stride * Sizeof.INT);

        long endGPUtime = System.nanoTime();

        double GPUtime = (endGPUtime - startGPUtime)/(1000000);
        System.out.println("GPU time: " + GPUtime + "ms \n");

        cuMemFree(deviceBoolArray);

        long startCPUtime = System.nanoTime();
        host_Control_Flow(num_iterations, num_arrays, array_stride, hostBool);
        long endCPUtime  = System.nanoTime();

        double CPUtime = (endCPUtime - startCPUtime)/(1000000);

        System.out.println("CPU time: " + CPUtime + "ms\n");

        int eq = isEqual(num_arrays * array_stride, hostDeviceOut, hostBool);
        if(eq == -1){
            System.out.println("Device and Host output the same");
        }
        else{
            System.out.println("Test failed.");
        }



    }

    private static void initialize(int num_arrays, int array_stride, int[] boolean_array){
        Random rand = new Random();
        for(int i=0; i < num_arrays * array_stride; i++){
            if(rand.nextInt(100) < 50) {
                boolean_array[i] = 0;
        }
            else{
                boolean_array[i] = 1;
            }
        }

    }

    private static int isEqual(int num,int[] array1, int[] array2){
        for(int i=0; i < num; i++){
            if(array1[i] != array2[i]){
                return i;
            }
        }
        return -1;
    }

    private static void host_Control_Flow(int n_iterations, int num_arrays, int array_stride, int[] boolean_array){
        for(int i=0; i< n_iterations; i++){

            for(int id = 0; id < num_arrays; id ++){

                int array_address = id * array_stride;

                int curr_val;
                int side_val;

                for(int j = 0; j < array_stride; j++){
                    curr_val = boolean_array[array_address + j];

                    if(curr_val == 0){
                        if(j > 0){
                            side_val = boolean_array[array_address + j - 1];
                            boolean_array[array_address + j - 1] = (side_val + 1) % 2;
                        }
                        else{
                            side_val = boolean_array[array_address + array_stride - 1];
                            boolean_array[array_address + array_stride - 1] = (side_val +1) % 2;
                        }
                    }
                    else{
                        if(j < array_stride - 1){
                            side_val = boolean_array[array_address + j + 1];
                            boolean_array[array_address + j + 1] = (side_val + 1) % 2;
                        }
                        else{
                            side_val = boolean_array[array_address];
                            boolean_array[array_address] = (side_val + 1) % 2;
                        }
                    }
                }
            }
        }
    }




    private static String preparePtxFile(String cuFileName) throws IOException
    {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1)
        {
            endIndex = cuFileName.length()-1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists())
        {
            return ptxFileName;
        }

        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");
        String command =
                "nvcc " + modelString + " -ptx "+
                        cuFile.getPath()+" -o "+ptxFileName;

        System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage =
                new String(toByteArray(process.getErrorStream()));
        String outputMessage =
                new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try
        {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
        }

        if (exitValue != 0)
        {
            System.out.println("nvcc process exitValue "+exitValue);
            System.out.println("errorMessage:\n"+errorMessage);
            System.out.println("outputMessage:\n"+outputMessage);
            throw new IOException(
                    "Could not create .ptx file: "+errorMessage);
        }

        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }

    /**
     * Fully reads the given InputStream and returns it as a byte array
     *
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(InputStream inputStream)
            throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }


}