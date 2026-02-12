package ai.qxotic.jota.examples.llama;

/**
 * JNI wrapper for optimized SwiGLU kernel.
 * Computes: out = gate * sigmoid(gate) * up
 */
public class SwigluKernel {
    
    static {
        // Load the JNI library
        try {
            System.loadLibrary("swiglu_jni");
        } catch (UnsatisfiedLinkError e) {
            // Try loading from the llamafile directory
            try {
                System.load("/home/mukel/Desktop/playground/llm4j/jota/jota-runtime-c/native/llamafile/libswiglu_jni.so");
            } catch (UnsatisfiedLinkError e2) {
                System.err.println("Failed to load SwiGLU kernel library: " + e2.getMessage());
            }
        }
    }
    
    /**
     * Returns true if the SwiGLU kernel library is available.
     */
    public static native boolean loadLibrary();
    
    /**
     * Compute SwiGLU activation.
     * 
     * @param gatePtr Pointer to gate matrix [dim, batch]
     * @param upPtr Pointer to up matrix [dim, batch]
     * @param outPtr Pointer to output matrix [dim, batch]
     * @param batch Batch size
     * @param dim Dimension (ffnDim)
     * @param gate_stride Stride for gate (batch)
     * @param up_stride Stride for up (batch)
     * @param out_stride Stride for output (batch)
     */
    public static native void compute(
        long gatePtr, long upPtr, long outPtr,
        int batch, int dim,
        int gate_stride, int up_stride, int out_stride
    );
    
    /**
     * Check if SwiGLU kernel is available.
     */
    public static boolean isAvailable() {
        try {
            return loadLibrary();
        } catch (UnsatisfiedLinkError e) {
            return false;
        }
    }
}
