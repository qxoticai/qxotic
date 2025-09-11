package com.llm4j.jota.cuda;

import com.llm4j.jota.FloatBinaryOperator;
import com.llm4j.jota.FloatUnaryOperator;
import com.llm4j.jota.memory.FloatOperations;
import com.llm4j.jota.memory.Memory;
import com.llm4j.jota.memory.impl.MemoryFactory;
import com.llm4j.jota.memory.MemoryView;
import com.llm4j.jota.memory.ScopedMemory;
import jcuda.Pointer;
import jcuda.driver.CUdeviceptr;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;
import jcuda.jcublas.cublasOperation;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;

public class CUDAFloatOperations implements FloatOperations<CUdeviceptr> {

    private static final CUDAFloatOperations INSTANCE = new CUDAFloatOperations();

    private final CUDAKernel unaryContiguousKernel;
    private final CUDAKernel unaryStridedKernel;
    private final CUDAKernel binaryContiguousKernel;
    private final CUDAKernel binaryStridedKernel;
    private final CUDAKernel scalarContiguousKernel;
    private final CUDAKernel scalarStridedKernel;

    private final CUDAKernel foldAllKernel;
    private final CUDAKernel foldKernel;
    private final CUDAKernel foldStridedKernel;
    private final CUDAKernel reduceContiguousKernel;
    private final CUDAKernel reduceStridedKernel;
    private final CUDAKernel fillWithUnaryScalarContiguousKernel;
    private final CUDAKernel fillWithUnaryScalarStridedKernel;

    private final CUDAContext cudaContext;

    public static FloatOperations<CUdeviceptr> instance() {
        return INSTANCE;
    }

    private CUDAFloatOperations() {
        this.unaryContiguousKernel = new CUDAKernel("/kernels.ptx", "elementwise_unary_contiguous");
        this.unaryStridedKernel = new CUDAKernel("/kernels.ptx", "elementwise_unary_strided");
        this.binaryContiguousKernel = new CUDAKernel("/kernels.ptx", "elementwise2_contiguous");
        this.binaryStridedKernel = new CUDAKernel("/kernels.ptx", "elementwise2_strided");
        this.scalarContiguousKernel = new CUDAKernel("/kernels.ptx", "elementwise2_scalar_contiguous");
        this.scalarStridedKernel = new CUDAKernel("/kernels.ptx", "elementwise2_scalar_strided");
        this.foldAllKernel = new CUDAKernel("/kernels.ptx", "fold_all_contiguous");
        this.foldKernel = new CUDAKernel("/kernels.ptx", "fold_contiguous");
        this.foldStridedKernel = new CUDAKernel("/kernels.ptx", "fold_strided");
        this.reduceContiguousKernel = new CUDAKernel("/kernels.ptx", "reduce_contiguous");
        this.reduceStridedKernel = new CUDAKernel("/kernels.ptx", "reduce_strided");
        this.fillWithUnaryScalarContiguousKernel = new CUDAKernel("/kernels.ptx", "fill_with_unary_scalar_contiguous");
        this.fillWithUnaryScalarStridedKernel = new CUDAKernel("/kernels.ptx", "fill_with_unary_scalar_strided");

        this.cudaContext = new CUDAContext(); // Initialize CUDAContext
    }

    private int getUnaryOpId(FloatUnaryOperator op) {
        if (op == FloatUnaryOperator.identity()) return 0;
        if (op == FloatUnaryOperator.square()) return 1;
        if (op == FloatUnaryOperator.exp()) return 2;
        throw new UnsupportedOperationException("Unsupported unary operator: " + op);
    }

    private int getBinaryOpId(FloatBinaryOperator op) {
        if (op == FloatBinaryOperator.sum()) return 0;
        if (op == FloatBinaryOperator.product()) return 1;
        if (op == FloatBinaryOperator.divide()) return 2;
        if (op == FloatBinaryOperator.subtract()) return 3;
        if (op == FloatBinaryOperator.max()) return 4;
        if (op == FloatBinaryOperator.min()) return 5;
        throw new UnsupportedOperationException("Unsupported binary operator: " + op);
    }

    @Override
    public void elementWise(MemoryView<CUdeviceptr> in, FloatUnaryOperator unaryOperator, MemoryView<CUdeviceptr> out) {
        int n = (int) in.shape().totalNumberOfElements();
        int opId = getUnaryOpId(unaryOperator);
        if (in.isContiguous() && out.isContiguous()) {
            launchUnaryContiguousKernel(in, out, n, opId);
        } else {
            launchUnaryStridedKernel(in, out, n, opId);
        }
    }

    @Override
    public void elementWise(float in, FloatUnaryOperator unaryOperator, MemoryView<CUdeviceptr> out) {
        int n = (int) out.shape().totalNumberOfElements();
        int opId = getUnaryOpId(unaryOperator);
        if (out.isContiguous()) {
            launchFillWithUnaryScalarContiguousKernel(in, out, n, opId);
        } else {
            launchFillWithUnaryScalarStridedKernel(in, out, n, opId);
        }
    }

    private void launchFillWithUnaryScalarContiguousKernel(float scalarIn, MemoryView<CUdeviceptr> out, int n, int opId) {
        Pointer kernelParameters = Pointer.to(
            Pointer.to(new float[]{scalarIn}),
            Pointer.to(out.memory().base().withByteOffset(out.byteOffset())),
            Pointer.to(new int[]{n}),
            Pointer.to(new int[]{opId})
        );
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        fillWithUnaryScalarContiguousKernel.launch(gridSize, 1, 1, blockSize, 1, 1, 0, kernelParameters);
    }

    private void launchFillWithUnaryScalarStridedKernel(float scalarIn, MemoryView<CUdeviceptr> out, int n, int opId) {
        int rank = out.shape().rank();
        Pointer kernelParameters = Pointer.to(
            Pointer.to(new float[]{scalarIn}),
            Pointer.to(out.memory().base().withByteOffset(out.byteOffset())),
            Pointer.to(out.shape().toArray()),
            Pointer.to(out.byteStrides()),
            Pointer.to(new int[]{n}),
            Pointer.to(new int[]{rank}),
            Pointer.to(new int[]{opId})
        );
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        fillWithUnaryScalarStridedKernel.launch(gridSize, 1, 1, blockSize, 1, 1, 0, kernelParameters);
    }

    @Override
    public void elementWise2(MemoryView<CUdeviceptr> left, FloatBinaryOperator binaryOperator, MemoryView<CUdeviceptr> right, MemoryView<CUdeviceptr> out) {
        int n = (int) left.shape().totalNumberOfElements();
        int opId = getBinaryOpId(binaryOperator);
        if (left.isContiguous() && right.isContiguous() && out.isContiguous()) {
            launchBinaryContiguousKernel(left, right, out, n, opId);
        } else {
            launchBinaryStridedKernel(left, right, out, n, opId);
        }
    }

    @Override
    public void elementWise2(MemoryView<CUdeviceptr> left, FloatBinaryOperator binaryOperator, float right, MemoryView<CUdeviceptr> out) {
        int n = (int) left.shape().totalNumberOfElements();
        int opId = getBinaryOpId(binaryOperator);
        if (left.isContiguous() && out.isContiguous()) {
            launchScalarContiguousKernel(left, right, out, n, opId);
        } else {
            launchScalarStridedKernel(left, right, out, n, opId);
        }
    }

    // Kernel Launchers

    private void launchUnaryContiguousKernel(MemoryView<CUdeviceptr> in, MemoryView<CUdeviceptr> out, int n, int opId) {
        Pointer kernelParameters = Pointer.to(
            Pointer.to(in.memory().base().withByteOffset(in.byteOffset())),
            Pointer.to(out.memory().base().withByteOffset(out.byteOffset())),
            Pointer.to(new int[]{n}),
            Pointer.to(new int[]{opId})
        );
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        unaryContiguousKernel.launch(gridSize, 1, 1, blockSize, 1, 1, 0, kernelParameters);
    }

    private void launchUnaryStridedKernel(MemoryView<CUdeviceptr> in, MemoryView<CUdeviceptr> out, int n, int opId) {
        int rank = in.shape().rank();
        Pointer kernelParameters = Pointer.to(
            Pointer.to(in.memory().base().withByteOffset(in.byteOffset())),
            Pointer.to(in.shape().toArray()),
            Pointer.to(in.byteStrides()),
            Pointer.to(out.memory().base().withByteOffset(out.byteOffset())),
            Pointer.to(out.shape().toArray()),
            Pointer.to(out.byteStrides()),
            Pointer.to(new int[]{n}),
            Pointer.to(new int[]{rank}),
            Pointer.to(new int[]{opId})
        );
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        unaryStridedKernel.launch(gridSize, 1, 1, blockSize, 1, 1, 0, kernelParameters);
    }

    private void launchBinaryContiguousKernel(MemoryView<CUdeviceptr> left, MemoryView<CUdeviceptr> right, MemoryView<CUdeviceptr> out, int n, int opId) {
        Pointer kernelParameters = Pointer.to(
            Pointer.to(left.memory().base().withByteOffset(left.byteOffset())),
            Pointer.to(right.memory().base().withByteOffset(right.byteOffset())),
            Pointer.to(out.memory().base().withByteOffset(out.byteOffset())),
            Pointer.to(new int[]{n}),
            Pointer.to(new int[]{opId})
        );
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        binaryContiguousKernel.launch(gridSize, 1, 1, blockSize, 1, 1, 0, kernelParameters);
    }

    private void launchBinaryStridedKernel(MemoryView<CUdeviceptr> left, MemoryView<CUdeviceptr> right, MemoryView<CUdeviceptr> out, int n, int opId) {
        int rank = left.shape().rank();
        Pointer kernelParameters = Pointer.to(
            Pointer.to(left.memory().base().withByteOffset(left.byteOffset())),
            Pointer.to(left.shape().toArray()),
            Pointer.to(left.byteStrides()),
            Pointer.to(right.memory().base().withByteOffset(right.byteOffset())),
            Pointer.to(right.shape().toArray()),
            Pointer.to(right.byteStrides()),
            Pointer.to(out.memory().base().withByteOffset(out.byteOffset())),
            Pointer.to(out.shape().toArray()),
            Pointer.to(out.byteStrides()),
            Pointer.to(new int[]{n}),
            Pointer.to(new int[]{rank}),
            Pointer.to(new int[]{opId})
        );
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        binaryStridedKernel.launch(gridSize, 1, 1, blockSize, 1, 1, 0, kernelParameters);
    }

    private void launchScalarContiguousKernel(MemoryView<CUdeviceptr> in, float scalar, MemoryView<CUdeviceptr> out, int n, int opId) {
        Pointer kernelParameters = Pointer.to(
            Pointer.to(in.memory().base().withByteOffset(in.byteOffset())),
            Pointer.to(new float[]{scalar}),
            Pointer.to(out.memory().base().withByteOffset(out.byteOffset())),
            Pointer.to(new int[]{n}),
            Pointer.to(new int[]{opId})
        );
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        scalarContiguousKernel.launch(gridSize, 1, 1, blockSize, 1, 1, 0, kernelParameters);
    }

    private void launchScalarStridedKernel(MemoryView<CUdeviceptr> in, float scalar, MemoryView<CUdeviceptr> out, int n, int opId) {
        int rank = in.shape().rank();
        Pointer kernelParameters = Pointer.to(
            Pointer.to(in.memory().base().withByteOffset(in.byteOffset())),
            Pointer.to(in.shape().toArray()),
            Pointer.to(in.byteStrides()),
            Pointer.to(new float[]{scalar}),
            Pointer.to(out.memory().base().withByteOffset(out.byteOffset())),
            Pointer.to(out.shape().toArray()),
            Pointer.to(out.byteStrides()),
            Pointer.to(new int[]{n}),
            Pointer.to(new int[]{rank}),
            Pointer.to(new int[]{opId})
        );
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;
        scalarStridedKernel.launch(gridSize, 1, 1, blockSize, 1, 1, 0, kernelParameters);
    }

    @Override
    public void fold(MemoryView<CUdeviceptr> in, FloatBinaryOperator binaryOperator, float initialValue, MemoryView<CUdeviceptr> out, int axis) {
        int rank = in.shape().rank();
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Invalid axis: " + axis);
        }

        if (in.isContiguous() && out.isContiguous()) {
            launchFoldContiguousKernel(in, binaryOperator, initialValue, out, axis);
        } else {
            launchFoldStridedKernel(in, binaryOperator, initialValue, out, axis);
        }
    }

    private void launchFoldContiguousKernel(MemoryView<CUdeviceptr> in, FloatBinaryOperator binaryOperator, float initialValue, MemoryView<CUdeviceptr> out, int axis) {
        int opId = getBinaryOpId(binaryOperator);
        int nOut = (int) out.shape().totalNumberOfElements();
        int rank = in.shape().rank();

        Pointer kernelParameters = Pointer.to(
            Pointer.to(in.memory().base().withByteOffset(in.byteOffset())),
            Pointer.to(out.memory().base().withByteOffset(out.byteOffset())),
            Pointer.to(new int[]{nOut}),
            Pointer.to(new int[]{rank}),
            Pointer.to(in.shape().toArray()),
            Pointer.to(new int[]{axis}),
            Pointer.to(new float[]{initialValue}),
            Pointer.to(new int[]{opId})
        );

        int blockSize = 256;
        int gridSize = (nOut + blockSize - 1) / blockSize;
        foldKernel.launch(gridSize, 1, 1, blockSize, 1, 1, 0, kernelParameters);
    }

    private void launchFoldStridedKernel(MemoryView<CUdeviceptr> in, FloatBinaryOperator binaryOperator, float initialValue, MemoryView<CUdeviceptr> out, int axis) {
        int opId = getBinaryOpId(binaryOperator);
        int nOut = (int) out.shape().totalNumberOfElements();
        int rank = in.shape().rank();

        Pointer kernelParameters = Pointer.to(
            Pointer.to(in.memory().base().withByteOffset(in.byteOffset())),
            Pointer.to(in.shape().toArray()),
            Pointer.to(in.byteStrides()),
            Pointer.to(out.memory().base().withByteOffset(out.byteOffset())),
            Pointer.to(out.shape().toArray()),
            Pointer.to(out.byteStrides()),
            Pointer.to(new int[]{nOut}),
            Pointer.to(new int[]{rank}),
            Pointer.to(new int[]{axis}),
            Pointer.to(new float[]{initialValue}),
            Pointer.to(new int[]{opId})
        );

        int blockSize = 256;
        int gridSize = (nOut + blockSize - 1) / blockSize;
        foldStridedKernel.launch(gridSize, 1, 1, blockSize, 1, 1, 0, kernelParameters);
    }

    @Override
    public void reduce(MemoryView<CUdeviceptr> in, FloatBinaryOperator binaryOperator, MemoryView<CUdeviceptr> out, int axis) {
        int rank = in.shape().rank();
        if (axis < 0 || axis >= rank) {
            throw new IllegalArgumentException("Invalid axis: " + axis);
        }

        if (in.isContiguous() && out.isContiguous()) {
            launchReduceContiguousKernel(in, binaryOperator, out, axis);
        } else {
            launchReduceStridedKernel(in, binaryOperator, out, axis);
        }
    }

    private void launchReduceContiguousKernel(MemoryView<CUdeviceptr> in, FloatBinaryOperator binaryOperator, MemoryView<CUdeviceptr> out, int axis) {
        int opId = getBinaryOpId(binaryOperator);
        int nOut = (int) out.shape().totalNumberOfElements();
        int rank = in.shape().rank();

        Pointer kernelParameters = Pointer.to(
            Pointer.to(in.memory().base().withByteOffset(in.byteOffset())),
            Pointer.to(out.memory().base().withByteOffset(out.byteOffset())),
            Pointer.to(new int[]{nOut}),
            Pointer.to(new int[]{rank}),
            Pointer.to(in.shape().toArray()),
            Pointer.to(new int[]{axis}),
            Pointer.to(new int[]{opId})
        );

        int blockSize = 256;
        int gridSize = (nOut + blockSize - 1) / blockSize;
        reduceContiguousKernel.launch(gridSize, 1, 1, blockSize, 1, 1, 0, kernelParameters);
    }

    private void launchReduceStridedKernel(MemoryView<CUdeviceptr> in, FloatBinaryOperator binaryOperator, MemoryView<CUdeviceptr> out, int axis) {
        int opId = getBinaryOpId(binaryOperator);
        int nOut = (int) out.shape().totalNumberOfElements();
        int rank = in.shape().rank();

        Pointer kernelParameters = Pointer.to(
            Pointer.to(in.memory().base().withByteOffset(in.byteOffset())),
            Pointer.to(in.shape().toArray()),
            Pointer.to(in.byteStrides()),
            Pointer.to(out.memory().base().withByteOffset(out.byteOffset())),
            Pointer.to(out.shape().toArray()),
            Pointer.to(out.byteStrides()),
            Pointer.to(new int[]{nOut}),
            Pointer.to(new int[]{rank}),
            Pointer.to(new int[]{axis}),
            Pointer.to(new int[]{opId})
        );

        int blockSize = 256;
        int gridSize = (nOut + blockSize - 1) / blockSize;
        reduceStridedKernel.launch(gridSize, 1, 1, blockSize, 1, 1, 0, kernelParameters);
    }

    @Override
    public float reduceAll(MemoryView<CUdeviceptr> in, FloatBinaryOperator binaryOperator) {
        if (!in.isContiguous()) {
            throw new UnsupportedOperationException("Strided reduceAll is not yet implemented for CUDA.");
        }
        int n = (int) in.shape().totalNumberOfElements();
        if (n == 0) {
            throw new IllegalArgumentException("Cannot reduce an empty tensor.");
        }
        // For reduceAll, the initial value is the first element of the tensor.
        // We can use foldAll by providing the first element as the initial value.
        // This requires copying the first element to the host, which is a small overhead.
        float initialValue = CUDAMemoryAccess.instance().readFloat(in.memory(), in.byteOffset());
        return foldAll(in.slice(0, 1, n), initialValue, binaryOperator);
    }

    @Override
    public float foldAll(MemoryView<CUdeviceptr> in, float initialValue, FloatBinaryOperator binaryOperator) {
        if (!in.isContiguous()) {
            throw new UnsupportedOperationException("Strided foldAll is not yet implemented for CUDA.");
        }

        int n = (int) in.shape().totalNumberOfElements();
        if (n == 0) {
            return initialValue;
        }
        int opId = getBinaryOpId(binaryOperator);

        int blockSize = 256; // A common, reasonable block size, must be a power of 2
        int gridSize = (n + blockSize - 1) / blockSize;

        // Allocate temporary buffer for partial results on the device
        try (ScopedMemory<CUdeviceptr> partialResultsDev = CUDAScopedMemoryAllocator.instance().allocateMemory(gridSize * 4L, 4)) {

            // --- First Kernel Launch: Partial Reduction ---
            Pointer kernelParams1 = Pointer.to(
                Pointer.to(in.memory().base().withByteOffset(in.byteOffset())),
                Pointer.to(partialResultsDev.base()),
                Pointer.to(new int[]{n}),
                Pointer.to(new float[]{initialValue}),
                Pointer.to(new int[]{opId})
            );
            int sharedMemBytes = blockSize * 4;
            foldAllKernel.launch(gridSize, 1, 1, blockSize, 1, 1, sharedMemBytes, kernelParams1);

            // If the first pass resulted in a single value, we can return it directly.
            if (gridSize == 1) {
                try (Arena arena = Arena.ofConfined()) {
                    MemorySegment hostSegment = arena.allocate(4);
                    CUDAMemoryOperations.instance().copyToNative(
                        partialResultsDev, 0,
                        MemoryFactory.ofMemorySegment(hostSegment), 0,
                        4L
                    );
                    return hostSegment.get(ValueLayout.JAVA_FLOAT, 0);
                }
            }

            // --- Second Kernel Launch: Final Reduction ---
            Pointer kernelParams2 = Pointer.to(
                Pointer.to(partialResultsDev.base()), // Input is the partial results
                Pointer.to(partialResultsDev.base()), // Output to the start of the same buffer
                Pointer.to(new int[]{gridSize}),      // n is now the number of partial results
                Pointer.to(new float[]{initialValue}),
                Pointer.to(new int[]{opId})
            );
            // Launch with a single block to complete the reduction
            foldAllKernel.launch(1, 1, 1, blockSize, 1, 1, sharedMemBytes, kernelParams2);

            // Copy the final single result back from the device
            try (Arena arena = Arena.ofConfined()) {
                MemorySegment hostSegment = arena.allocate(4);
                CUDAMemoryOperations.instance().copyToNative(
                    partialResultsDev, 0, // Read from the start of the buffer
                    MemoryFactory.ofMemorySegment(hostSegment), 0,
                    4L
                );
                return hostSegment.get(ValueLayout.JAVA_FLOAT, 0);
            }
        }
    }

    @Override
    public void matrixMultiply(MemoryView<CUdeviceptr> left, MemoryView<CUdeviceptr> right, MemoryView<CUdeviceptr> out) {
        // Validate shapes
        if (left.shape().rank() != 2 || right.shape().rank() != 2 || out.shape().rank() != 2) {
            throw new IllegalArgumentException("Matrix multiplication requires 2D tensors.");
        }
        if (left.shape().dimension(1) != right.shape().dimension(0)) {
            throw new IllegalArgumentException("Inner dimensions must match for matrix multiplication: " +
                                               left.shape().dimension(1) + " != " + right.shape().dimension(0));
        }
        if (left.shape().dimension(0) != out.shape().dimension(0) ||
            right.shape().dimension(1) != out.shape().dimension(1)) {
            throw new IllegalArgumentException("Output shape mismatch: Expected (" +
                                               left.shape().dimension(0) + ", " + right.shape().dimension(1) +
                                               "), got " + out.shape());
        }

        // cuBLAS requires contiguous, column-major matrices.
        // Since Jota uses row-major, we conceptually transpose the operation: C = A @ B becomes C^T = B^T @ A^T
        // So, we call SGEMM with B as the first matrix and A as the second, both transposed.
        if (!left.isContiguous() || !right.isContiguous() || !out.isContiguous()) {
            throw new UnsupportedOperationException("Strided matrix multiplication is not yet supported. " +
                                                    "Please ensure all input and output views are contiguous.");
        }

        int M = (int) left.shape().dimension(0); // Rows of A
        int K = (int) left.shape().dimension(1); // Cols of A, Rows of B
        int N = (int) right.shape().dimension(1); // Cols of B

        // Alpha and Beta values for C = alpha * A * B + beta * C
        float alpha = 1.0f;
        float beta = 0.0f;

        // Pointers to device memory
        CUdeviceptr A_dev = left.memory().base().withByteOffset(left.byteOffset());
        CUdeviceptr B_dev = right.memory().base().withByteOffset(right.byteOffset());
        CUdeviceptr C_dev = out.memory().base().withByteOffset(out.byteOffset());

        // Perform C = A @ B (row-major) using cuBLAS's column-major SGEMM
        // This translates to C^T = B^T @ A^T
        // Parameters for cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc)
        // Here, A in SGEMM is B_dev, B in SGEMM is A_dev, C in SGEMM is C_dev
        // m = N (rows of B^T / cols of B)
        // n = M (cols of A^T / rows of A)
        // k = K (common dimension)
        // lda = N (leading dimension of B^T, which is cols of B)
        // ldb = K (leading dimension of A^T, which is cols of A)
        // ldc = N (leading dimension of C^T, which is cols of C)

        JCublas2.cublasSgemm(
            cudaContext.getCublasHandle(),
            CUBLAS_OP_T, // transa: Transpose B (right matrix)
            CUBLAS_OP_T, // transb: Transpose A (left matrix)
            N,           // m: rows of op(A) (B^T) and C (C^T)
            M,           // n: cols of op(B) (A^T) and C (C^T)
            K,           // k: common dimension
            Pointer.to(new float[]{alpha}),
            B_dev,       // A: Pointer to B (right matrix)
            N,           // lda: leading dimension of B (cols of B)
            A_dev,       // B: Pointer to A (left matrix)
            K,           // ldb: leading dimension of A (cols of A)
            Pointer.to(new float[]{beta}),
            C_dev,       // C: Pointer to C (output matrix)
            N            // ldc: leading dimension of C (cols of C)
        );
    }
}