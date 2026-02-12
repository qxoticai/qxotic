package ai.qxotic.jota.examples.llama;

/**
 * JNI wrapper for fused kernels library.
 * Provides high-performance fused operations for Llama inference.
 */
public class FusedKernels {
    
    static {
        // Load the JNI library
        try {
            System.loadLibrary("fused_kernels_jni");
        } catch (UnsatisfiedLinkError e) {
            // Try loading from the llamafile directory
            try {
                System.load("/home/mukel/Desktop/playground/llm4j/jota/jota-runtime-c/native/llamafile/libfused_kernels_jni.so");
            } catch (UnsatisfiedLinkError e2) {
                System.err.println("Failed to load fused kernels library: " + e2.getMessage());
            }
        }
    }
    
    /**
     * Returns true if the fused kernels library is available.
     */
    public static native boolean loadLibrary();
    
    /**
     * Fused RMSNorm + QKV projections.
     * Computes Q, K, V in a single kernel call.
     */
    public static native void fusedRmsnormQkv(
        long xPtr, long attnNormPtr,
        long wqPtr, long wkPtr, long wvPtr,
        long qPtr, long kPtr, long vPtr,
        int batch, int dim, int kvDim,
        float rmsEps, int xStride, int qStride, int kStride, int vStride
    );
    
    /**
     * Fused RMSNorm + FFN projections + SwiGLU.
     * Computes hidden state in a single kernel call.
     */
    public static native void fusedRmsnormFfn(
        long xPtr, long ffnNormPtr,
        long wGatePtr, long wUpPtr,
        long hiddenPtr,
        int batch, int dim, int ffnDim,
        float rmsEps, int xStride, int hiddenStride
    );
    
    /**
     * Check if fused kernels are available.
     */
    public static boolean isAvailable() {
        try {
            return loadLibrary();
        } catch (UnsatisfiedLinkError e) {
            return false;
        }
    }
}
