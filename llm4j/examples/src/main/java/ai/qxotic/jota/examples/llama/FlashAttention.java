package ai.qxotic.jota.examples.llama;

/**
 * JNI wrapper for Flash Attention library.
 * Provides memory-efficient attention computation using tiling.
 */
public class FlashAttention {
    
    static {
        // Load the JNI library
        try {
            System.loadLibrary("flash_attention_jni");
        } catch (UnsatisfiedLinkError e) {
            // Try loading from the llamafile directory
            try {
                System.load("/home/mukel/Desktop/playground/llm4j/jota/jota-runtime-c/native/llamafile/libflash_attention_jni.so");
            } catch (UnsatisfiedLinkError e2) {
                System.err.println("Failed to load Flash Attention library: " + e2.getMessage());
            }
        }
    }
    
    /**
     * Returns true if the Flash Attention library is available.
     */
    public static native boolean loadLibrary();
    
    /**
     * Compute Flash Attention.
     * 
     * @param qPtr Pointer to Q matrix [n_heads, seq_q, head_dim]
     * @param kPtr Pointer to K matrix [n_heads, seq_kv, head_dim]
     * @param vPtr Pointer to V matrix [n_heads, seq_kv, head_dim]
     * @param outputPtr Pointer to output matrix [n_heads, seq_q, head_dim]
     * @param n_heads Number of attention heads
     * @param seq_q Query sequence length
     * @param seq_kv Key/Value sequence length
     * @param head_dim Head dimension
     * @param scale Attention scale (typically 1/sqrt(head_dim))
     */
    public static native void compute(
        long qPtr, long kPtr, long vPtr,
        long outputPtr,
        int n_heads, int seq_q, int seq_kv, int head_dim,
        float scale
    );
    
    /**
     * Check if Flash Attention is available.
     */
    public static boolean isAvailable() {
        try {
            return loadLibrary();
        } catch (UnsatisfiedLinkError e) {
            return false;
        }
    }
}
