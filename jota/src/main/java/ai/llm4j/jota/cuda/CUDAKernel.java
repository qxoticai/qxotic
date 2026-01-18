package ai.llm4j.jota.cuda;

import java.io.IOException;
import java.io.InputStream;
import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

public class CUDAKernel implements AutoCloseable {

    private final CUmodule module;
    private final CUfunction function;

    public CUDAKernel(String ptxPath, String kernelName) {
        try (InputStream ptxInputStream = CUDAKernel.class.getResourceAsStream(ptxPath)) {
            if (ptxInputStream == null) {
                throw new IOException("PTX file not found: " + ptxPath);
            }
            byte[] ptxData = ptxInputStream.readAllBytes();

            module = new CUmodule();
            JCudaDriver.cuModuleLoadData(module, ptxData);
            function = new CUfunction();
            JCudaDriver.cuModuleGetFunction(function, module, kernelName);
        } catch (IOException e) {
            throw new RuntimeException("Failed to load CUDA kernel", e);
        }
    }

    public void launch(
            int gridDimX,
            int gridDimY,
            int gridDimZ,
            int blockDimX,
            int blockDimY,
            int blockDimZ,
            int sharedMemBytes,
            Pointer kernelParameters) {
        JCudaDriver.cuLaunchKernel(
                function,
                gridDimX,
                gridDimY,
                gridDimZ,
                blockDimX,
                blockDimY,
                blockDimZ,
                sharedMemBytes,
                null, // hStream
                kernelParameters,
                null // Extra
                );
        JCudaDriver.cuCtxSynchronize();
    }

    @Override
    public void close() {
        JCudaDriver.cuModuleUnload(module);
    }
}
