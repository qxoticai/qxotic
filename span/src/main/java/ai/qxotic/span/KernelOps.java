package ai.qxotic.span;

/**
 * Core computational interface for tensor operations in the LLaMA model implementation. This
 * interface defines the fundamental operations required for transformer model inference, including
 * attention mechanisms, matrix operations, and various tensor manipulations.
 *
 * <p>A key design principle of this interface is the "potato tensor" approach - tensors are treated
 * as opaque data structures that cannot be read from or inspected. This deliberate design choice: -
 * Forces all operations to be self-contained within the implementation - Enables efficient
 * zero-copy implementations on accelerators (GPU, etc.) - Prevents accidental data transfers
 * between host and device memory - Maintains clean abstraction boundaries between compute and data
 * management
 *
 * @param <V> The type of vector/tensor spans, must extend FloatSpan
 * @param <M> The type of matrix views, must extend FloatMatrixView
 */
public interface KernelOps<V extends FloatSpan, M extends FloatMatrixView> {

    /**
     * Applies a unary operation to each element in the input span.
     *
     * @param span The input tensor
     * @param mapper The operation to apply to each element
     * @param out The output tensor (must be same size as input)
     */
    void elementWise(V span, FloatUnaryOperator mapper, V out);

    /**
     * Applies a binary operation between corresponding elements of two spans.
     *
     * @param span First input tensor
     * @param other Second input tensor
     * @param combiner The operation to apply between corresponding elements
     * @param out The output tensor (must be same size as inputs)
     */
    void map2(V span, V other, FloatBinaryOperator combiner, V out);

    /**
     * Copies content from one span to another. This operation maintains the "potato tensor"
     * principle by not exposing the actual data.
     *
     * @param span Source tensor
     * @param out Destination tensor (must be same size as source)
     */
    void copyTo(V span, V out);

    /**
     * Copies content from input span to output span with strided access pattern. Allows for
     * non-contiguous memory access patterns in the output tensor.
     *
     * @param in Input tensor to copy from
     * @param out Output tensor to copy to
     * @param outOffset Starting offset in the output tensor
     * @param outElementStride Stride between elements in the output tensor
     * @throws IllegalArgumentException if output tensor is too small for the strided copy
     */
    void copyToStrided(V in, V out, long outOffset, long outElementStride);

    /**
     * Computes the softmax function over the input span. softmax(x_i) = exp(x_i) / sum(exp(x_j))
     *
     * @param span Input tensor
     * @param out Output tensor (must be same size as input)
     */
    void softMax(V span, V out);

    /**
     * Applies RMSNorm (Root Mean Square Layer Normalization). out = x * w / sqrt(mean(x^2) + eps)
     *
     * @param span Input tensor
     * @param weight Weight tensor for the normalization
     * @param rmsNormEps Small epsilon value to prevent division by zero
     * @param out Output tensor (must be same size as input)
     */
    void rmsNorm(V span, V weight, float rmsNormEps, V out);

    /**
     * Multiplies a matrix by a vector: out = matrix @ vector The operation assumes compatible
     * dimensions: [rows, cols] @ [cols] = [rows]
     *
     * @param matrix Input matrix of shape [rows, cols]
     * @param vector Input vector of length cols
     * @param out Output vector of length rows
     */
    void matrixVectorMultiply(M matrix, V vector, V out);

    /**
     * Performs matrix multiplication between two matrices: out = a @ b^T The operation assumes
     * row-major order for all matrices.
     *
     * <p>{@code A=[R, K] B=[K, C]^T C=[R, C]}
     *
     * @param R Number of rows in matrix a and output matrix
     * @param C Number of columns in matrix b and output matrix
     * @param K Number of columns in matrix a and rows in matrix b
     * @param a First input matrix of shape [R, K]
     * @param b Second input matrix of shape [K, C] (transposed)
     * @param out Output matrix of shape [R, C]
     * @throws IllegalArgumentException if matrix dimensions are incompatible
     */
    default void matrixMultiply(long R, long C, long K, M a, M b, M out) {
        assert R <= a.rows();
        assert K <= a.cols();

        assert C <= b.rows();
        assert K <= b.cols();

        assert R <= out.rows();
        assert C <= out.cols();

        gemmRowMajor(
                R,
                C,
                K,
                a.innerSpan(),
                a.startOffset(),
                a.rowStride(),
                b.innerSpan(),
                b.startOffset(),
                b.rowStride(),
                out.innerSpan(),
                out.startOffset(),
                out.rowStride());
    }

    /**
     * Performs general matrix multiplication (GEMM) in row-major order. This is equivalent to the
     * BLAS GEMM operation in column-major order with parameters: {@code GEMM("T", "N", C, R, K, 1f,
     * B, B_offset, ldb, A, A_offset, lda, 0f, C, C_offset, ldc)}
     *
     * <p>Rather use {@link #matrixMultiply(long, long, long, FloatMatrixView, FloatMatrixView,
     * FloatMatrixView)} which provide more simple interface.
     */
    void gemmRowMajor(
            long R,
            long C,
            long K,
            FloatSpan a,
            long aOffset,
            long aRowStride, // [R, K]
            FloatSpan b,
            long bOffset,
            long bRowStride, // [K, C]^T
            FloatSpan out,
            long outOffset,
            long outRowStride); // [R, C]

    /**
     * Applies rotary position embeddings to the input tensor. Supports both original and NeoX-style
     * implementations.
     *
     * @param neoxStyle If true, uses NeoX-style rotary embeddings
     * @param span Input tensor
     * @param freqReal Real components of rotation frequencies
     * @param freqImag Imaginary components of rotation frequencies
     * @param position Current sequence position
     * @param numberOfHeads Number of attention heads
     * @param headSize Size of each attention head
     * @param out Output tensor (must be same size as input)
     */
    void rotate(
            boolean neoxStyle,
            V span,
            V freqReal,
            V freqImag,
            int position,
            int numberOfHeads,
            int headSize,
            V out);

    /**
     * Fills the output tensor with a constant value.
     *
     * @param value Value to fill with
     * @param out Output tensor to fill
     */
    void fill(float value, V out);

    /**
     * Scales all elements in the input tensor by a constant value.
     *
     * @param span Input tensor
     * @param value Scaling factor
     * @param out Output tensor (must be same size as input)
     */
    void scale(V span, float value, V out);

    /**
     * Adds two tensors element-wise.
     *
     * @param span First input tensor
     * @param other Second input tensor
     * @param out Output tensor (must be same size as inputs)
     */
    default void add(V span, V other, V out) {
        map2(span, other, FloatBinaryOperator.SUM, out);
    }

    /**
     * Multiplies two tensors element-wise.
     *
     * @param span First input tensor
     * @param other Second input tensor
     * @param out Output tensor (must be same size as inputs)
     */
    default void multiply(V span, V other, V out) {
        map2(span, other, FloatBinaryOperator.MUL, out);
    }
}
