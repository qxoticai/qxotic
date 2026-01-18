package ai.qxotic.model.llm.llama;

import ai.qxotic.format.gguf.GGMLType;
import ai.qxotic.span.DirectAccessOps;
import ai.qxotic.span.FloatMatrixView;
import ai.qxotic.span.FloatSpan;
// import dev.ludovic.netlib.blas.BLAS;
// import dev.ludovic.netlib.blas.BLAS;
import java.nio.ByteOrder;
import java.util.function.Function;
import jdk.incubator.vector.*;

// import net.dedekind.blas.Blas;
// import net.dedekind.blas.Blas;
// import org.netlib.blas.Sgemm;

public class PanamaKernelOps extends BaseKernelOps {

    static final int VECTOR_BIT_SIZE =
            Integer.getInteger(
                    "ai.qxotic.VectorBitSize", VectorShape.preferredShape().vectorBitSize());
    static final boolean USE_VECTOR_API = VECTOR_BIT_SIZE != 0;

    static final VectorSpecies<Float> F_SPECIES;
    static final VectorSpecies<Integer> I_SPECIES;
    static final VectorSpecies<Short> S_SPECIES_HALF;

    static {
        if (USE_VECTOR_API) {
            F_SPECIES = VectorShape.forBitSize(VECTOR_BIT_SIZE).withLanes(float.class);
            I_SPECIES = F_SPECIES.withLanes(int.class);
            S_SPECIES_HALF =
                    VectorShape.forBitSize(F_SPECIES.vectorBitSize() / 2).withLanes(short.class);
            assert F_SPECIES.length() == S_SPECIES_HALF.length();
        } else {
            F_SPECIES = null;
            I_SPECIES = null;
            S_SPECIES_HALF = null;
        }
    }

    public PanamaKernelOps(Function<FloatSpan, DirectAccessOps<FloatSpan>> directAccess) {
        super(directAccess);
    }

    @Override
    public void matrixVectorMultiply(FloatMatrixView matrix, FloatSpan vector, FloatSpan out) {
        long rows = matrix.rows();
        long cols = matrix.cols();
        if (USE_VECTOR_API && vector instanceof ArraySpan arrayVector) {
            var outAccess = directAccess.apply(out);

            Math.multiplyExact(rows - 1, cols); //
            int intCols = Math.toIntExact(cols);

            var innerSpan = matrix.innerSpan();

            if (innerSpan instanceof ArraySpan arraySpan) {
                Parallel.parallelForLong(
                        0,
                        rows,
                        i -> {
                            float result =
                                    vectorDot(
                                            arraySpan,
                                            matrix.rowOffset(i),
                                            arrayVector,
                                            0,
                                            intCols);
                            outAccess.setElementAt(out, i, result);
                        });
                return;
            }

            if (innerSpan instanceof Q8_0Span q8_0Matrix) {
                Parallel.parallelForLong(
                        0,
                        rows,
                        i -> {
                            float result =
                                    vectorDot(
                                            q8_0Matrix,
                                            matrix.rowOffset(i),
                                            arrayVector,
                                            0,
                                            intCols);
                            outAccess.setElementAt(out, i, result);
                        });
                return;
            }
            if (innerSpan instanceof Q4_0Span q4_0Matrix) {
                Parallel.parallelForLong(
                        0,
                        rows,
                        i -> {
                            float result =
                                    vectorDot(
                                            q4_0Matrix,
                                            matrix.rowOffset(i),
                                            arrayVector,
                                            0,
                                            intCols);
                            outAccess.setElementAt(out, i, result);
                        });
                return;
            }
            if (innerSpan instanceof Q4_1Span q4_1Matrix) {
                Parallel.parallelForLong(
                        0,
                        rows,
                        i -> {
                            float result =
                                    vectorDot(
                                            q4_1Matrix,
                                            matrix.rowOffset(i),
                                            arrayVector,
                                            0,
                                            intCols);
                            outAccess.setElementAt(out, i, result);
                        });
                return;
            }
            if (innerSpan instanceof BF16Span bf16Matrix) {
                Parallel.parallelForLong(
                        0,
                        rows,
                        i -> {
                            float result =
                                    vectorDot(
                                            bf16Matrix,
                                            matrix.rowOffset(i),
                                            arrayVector,
                                            0,
                                            intCols);
                            outAccess.setElementAt(out, i, result);
                        });
                return;
            }
        }
        super.matrixVectorMultiply(matrix, vector, out);
    }

    public void daMatMulBF16(
            int rows, int cols, int batchSize, FloatSpan left, FloatSpan right, FloatSpan result) {
        daMatMulBF160(
                rows, cols, batchSize, (BF16Span) left, (ArraySpan) right, (ArraySpan) result);
    }

    void daMatMulBF160(
            int rows, int cols, int batchSize, BF16Span thiz, ArraySpan that, ArraySpan out) {
        if (cols % F_SPECIES.length() != 0) {
            throw new UnsupportedOperationException(
                    "TODO: implement tail, columns is not a multiple of " + F_SPECIES.length());
        }
        var outAccess = directAccess.apply(out);
        int rowStep = 4;
        Parallel.parallelForLong(
                0,
                rows / rowStep,
                rXX -> {
                    int r = (int) rXX * rowStep;
                    Parallel.parallelFor(
                            0,
                            batchSize,
                            b -> { // );for (int b = 0; b < batchSize; ++b) {
                                FloatVector result0 = FloatVector.zero(F_SPECIES);
                                FloatVector result1 = FloatVector.zero(F_SPECIES);
                                FloatVector result2 = FloatVector.zero(F_SPECIES);
                                FloatVector result3 = FloatVector.zero(F_SPECIES);

                                for (int c = 0; c < cols; c += F_SPECIES.length()) {
                                    FloatVector thatVector =
                                            getFloatVector(F_SPECIES, that, b * cols + c);
                                    result0 =
                                            ShortVector.fromMemorySegment(
                                                            S_SPECIES_HALF,
                                                            thiz.memorySegment,
                                                            (r * cols + c) * (long) BFloat16.BYTES,
                                                            ByteOrder.LITTLE_ENDIAN)
                                                    .castShape(I_SPECIES, 0) // (int) vi
                                                    .lanewise(VectorOperators.LSHL, 16) // vi <<= 16
                                                    .reinterpretAsFloats()
                                                    .fma(thatVector, result0);
                                    result1 =
                                            ShortVector.fromMemorySegment(
                                                            S_SPECIES_HALF,
                                                            thiz.memorySegment,
                                                            ((r + 1) * cols + c)
                                                                    * (long) BFloat16.BYTES,
                                                            ByteOrder.LITTLE_ENDIAN)
                                                    .castShape(I_SPECIES, 0) // (int) vi
                                                    .lanewise(VectorOperators.LSHL, 16) // vi <<= 16
                                                    .reinterpretAsFloats()
                                                    .fma(thatVector, result1);
                                    result2 =
                                            ShortVector.fromMemorySegment(
                                                            S_SPECIES_HALF,
                                                            thiz.memorySegment,
                                                            ((r + 2) * cols + c)
                                                                    * (long) BFloat16.BYTES,
                                                            ByteOrder.LITTLE_ENDIAN)
                                                    .castShape(I_SPECIES, 0) // (int) vi
                                                    .lanewise(VectorOperators.LSHL, 16) // vi <<= 16
                                                    .reinterpretAsFloats()
                                                    .fma(thatVector, result2);
                                    result3 =
                                            ShortVector.fromMemorySegment(
                                                            S_SPECIES_HALF,
                                                            thiz.memorySegment,
                                                            ((r + 3) * cols + c)
                                                                    * (long) BFloat16.BYTES,
                                                            ByteOrder.LITTLE_ENDIAN)
                                                    .castShape(I_SPECIES, 0) // (int) vi
                                                    .lanewise(VectorOperators.LSHL, 16) // vi <<= 16
                                                    .reinterpretAsFloats()
                                                    .fma(thatVector, result3);
                                }

                                outAccess.setElementAt(
                                        out,
                                        b * rows + r + 0,
                                        result0.reduceLanes(VectorOperators.ADD));
                                outAccess.setElementAt(
                                        out,
                                        b * rows + r + 1,
                                        result1.reduceLanes(VectorOperators.ADD));
                                outAccess.setElementAt(
                                        out,
                                        b * rows + r + 2,
                                        result2.reduceLanes(VectorOperators.ADD));
                                outAccess.setElementAt(
                                        out,
                                        b * rows + r + 3,
                                        result3.reduceLanes(VectorOperators.ADD));

                                // out[b][r] = ;
                            });
                });
    }

    public void daMatMulArray(
            int rows, int cols, int batchSize, FloatSpan left, FloatSpan right, FloatSpan result) {
        daMatMulArray0(
                rows, cols, batchSize, (ArraySpan) left, (ArraySpan) right, (ArraySpan) result);
    }

    void daMatMulArray0(
            int rows, int cols, int batchSize, ArraySpan thiz, ArraySpan that, ArraySpan out) {
        //        if (rows % 4 != 0) {
        //            throw new UnsupportedOperationException("TODO: implement tail, columns is not
        // a multiple of " + F_SPECIES.length());
        //        }
        var arrayAccess = directAccess.apply(out);
        int rowStep = 4;
        Parallel.parallelForLong(
                0,
                rows / rowStep,
                rXX -> {
                    int r = (int) rXX * rowStep;
                    Parallel.parallelFor(
                            0,
                            batchSize,
                            b -> { // );for (int b = 0; b < batchSize; ++b) {
                                FloatVector result0 = FloatVector.zero(F_SPECIES);
                                FloatVector result1 = FloatVector.zero(F_SPECIES);
                                FloatVector result2 = FloatVector.zero(F_SPECIES);
                                FloatVector result3 = FloatVector.zero(F_SPECIES);

                                int c = 0;
                                int upperBound = F_SPECIES.loopBound(cols);
                                for (; c < upperBound; c += F_SPECIES.length()) {
                                    FloatVector thatVector =
                                            getFloatVector(F_SPECIES, that, b * cols + c);
                                    result0 =
                                            FloatVector.fromArray(
                                                            F_SPECIES,
                                                            thiz.values,
                                                            thiz.offset + ((r + 0) * cols + c))
                                                    .fma(thatVector, result0);
                                    result1 =
                                            FloatVector.fromArray(
                                                            F_SPECIES,
                                                            thiz.values,
                                                            thiz.offset + ((r + 2) * cols + c))
                                                    .fma(thatVector, result0);
                                    result2 =
                                            FloatVector.fromArray(
                                                            F_SPECIES,
                                                            thiz.values,
                                                            thiz.offset + ((r + 3) * cols + c))
                                                    .fma(thatVector, result0);
                                    result3 =
                                            FloatVector.fromArray(
                                                            F_SPECIES,
                                                            thiz.values,
                                                            thiz.offset + ((r + 4) * cols + c))
                                                    .fma(thatVector, result0);
                                }

                                float sum0 = result0.reduceLanes(VectorOperators.ADD);
                                float sum1 = result1.reduceLanes(VectorOperators.ADD);
                                float sum2 = result2.reduceLanes(VectorOperators.ADD);
                                float sum3 = result3.reduceLanes(VectorOperators.ADD);

                                // tail
                                while (c < cols) {
                                    sum0 +=
                                            arrayAccess.getElementAt(that, b * cols + c)
                                                    * arrayAccess.getElementAt(
                                                            thiz, ((r + 0) * cols + c));
                                    sum1 +=
                                            arrayAccess.getElementAt(that, b * cols + c)
                                                    * arrayAccess.getElementAt(
                                                            thiz, ((r + 1) * cols + c));
                                    sum2 +=
                                            arrayAccess.getElementAt(that, b * cols + c)
                                                    * arrayAccess.getElementAt(
                                                            thiz, ((r + 2) * cols + c));
                                    sum3 +=
                                            arrayAccess.getElementAt(that, b * cols + c)
                                                    * arrayAccess.getElementAt(
                                                            thiz, ((r + 3) * cols + c));
                                    ++c;
                                }

                                arrayAccess.setElementAt(out, b * rows + r + 0, sum0);
                                arrayAccess.setElementAt(out, b * rows + r + 1, sum1);
                                arrayAccess.setElementAt(out, b * rows + r + 2, sum2);
                                arrayAccess.setElementAt(out, b * rows + r + 3, sum3);
                            });
                });

        for (int rLoop = rows / 4 * 4; rLoop < rows; ++rLoop) {
            int r = rLoop;
            Parallel.parallelFor(
                    0,
                    batchSize,
                    b -> { // );for (int b = 0; b < batchSize; ++b) {
                        FloatVector result0 = FloatVector.zero(F_SPECIES);
                        int c = 0;
                        int upperBound = F_SPECIES.loopBound(cols);
                        for (; c < upperBound; c += F_SPECIES.length()) {
                            FloatVector thatVector = getFloatVector(F_SPECIES, that, b * cols + c);
                            result0 =
                                    FloatVector.fromArray(
                                                    F_SPECIES,
                                                    thiz.values,
                                                    thiz.offset + ((r + 0) * cols + c))
                                            .fma(thatVector, result0);
                        }
                        float sum0 = result0.reduceLanes(VectorOperators.ADD);
                        // tail
                        while (c < cols) {
                            sum0 +=
                                    arrayAccess.getElementAt(that, b * cols + c)
                                            * arrayAccess.getElementAt(thiz, ((r + 0) * cols + c));
                            ++c;
                        }
                        arrayAccess.setElementAt(out, b * rows + r + 0, sum0);
                    });
        }
    }

    //    @Override
    //    public void matrixMultiply(FloatMatrixView a, int batchSize, FloatMatrixView b,
    // FloatMatrixView out) {
    //        if (batchSize == 1) {
    //            matrixVectorMultiply(a, b.row(0), out.row(0));
    //            return;
    //        }
    //
    //        long rows = a.rows();
    //        long cols = b.cols();
    //        if (USE_VECTOR_API && b.innerSpan() instanceof ArraySpan arrayVector) {
    //            var outAccess = directAccess.apply(out.innerSpan());
    //
    //            Math.multiplyExact(rows - 1, cols);
    //            int intCols = Math.toIntExact(cols);
    //
    //            var innerSpan = a.innerSpan();
    //
    //            if (innerSpan instanceof ArraySpan arraySpan) {
    //                Parallel.parallelForLong(0, rows, r -> {
    //                    for (int k = 0; k < batchSize; ++k) {
    //                        float result = vectorDot(arraySpan, a.rowOffset(r), arrayVector, (int)
    // b.rowOffset(k), intCols);
    //                        outAccess.setElementAt(out.innerSpan(), out.rowOffset(k) + r, result);
    //                    }
    //                });
    //                return;
    //            }
    //
    //            if (innerSpan instanceof Q8_0Span q8_0Matrix) {
    //                Parallel.parallelForLong(0, rows, r -> {
    //                    for (int k = 0; k < batchSize; ++k) {
    //                        float result = vectorDot(q8_0Matrix, a.rowOffset(r), arrayVector,
    // (int) b.rowOffset(k), intCols);
    //                        outAccess.setElementAt(out.innerSpan(), out.rowOffset(k) + r, result);
    //                    }
    //                });
    //                return;
    //            }
    //
    //            if (innerSpan instanceof Q4_0Span q8_4Matrix) {
    //                Parallel.parallelForLong(0, rows, r -> {
    //                    for (int k = 0; k < batchSize; ++k) {
    //                        float result = vectorDot(q8_4Matrix, a.rowOffset(r), arrayVector,
    // (int) b.rowOffset(k), intCols);
    //                        outAccess.setElementAt(out.innerSpan(), out.rowOffset(k) + r, result);
    //                    }
    //                });
    //                return;
    //            }
    //
    //            if (innerSpan instanceof Q4_1Span q4_1Matrix) {
    //                Parallel.parallelForLong(0, rows, r -> {
    //                    for (int k = 0; k < batchSize; ++k) {
    //                        float result = vectorDot(q4_1Matrix, a.rowOffset(r), arrayVector,
    // (int) b.rowOffset(k), intCols);
    //                        outAccess.setElementAt(out.innerSpan(), out.rowOffset(k) + r, result);
    //                    }
    //                });
    //                return;
    //            }
    //
    //            if (innerSpan instanceof BF16Span bf16Span) {
    //                Parallel.parallelForLong(0, rows, r -> {
    //                    for (int k = 0; k < batchSize; ++k) {
    //                        float result = vectorDot(bf16Span, a.rowOffset(r), arrayVector, (int)
    // b.rowOffset(k), intCols);
    //                        outAccess.setElementAt(out.innerSpan(), out.rowOffset(k) + r, result);
    //                    }
    //                });
    //                return;
    //            }
    //        }
    //        super.matrixMultiply(a, batchSize, b, out);
    //    }

    // static final Blas BLAS = Blas.getInstance(false);

    @Override
    public void gemmRowMajor(
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
            long outRowStride) { // [R, C]

        var outAccess = directAccess.apply(out);

        if (a instanceof ArraySpan aArraySpan) {

            if (b instanceof Q4_0Span bQ4_0Span) {
                Parallel.parallelForLong(
                        0,
                        R,
                        r -> {
                            Parallel.parallelForLong(
                                    0,
                                    C,
                                    c -> {
                                        float result =
                                                vectorDot(
                                                        bQ4_0Span,
                                                        bOffset + bRowStride * c,
                                                        aArraySpan,
                                                        (int) (aOffset + aRowStride * r),
                                                        (int) K);
                                        outAccess.setElementAt(
                                                out, outOffset + outRowStride * r + c, result);
                                    });
                        });
                return;
            }

            if (b instanceof Q8_0Span bQ8_0Span) {
                Parallel.parallelForLong(
                        0,
                        R,
                        r -> {
                            Parallel.parallelForLong(
                                    0,
                                    C,
                                    c -> {
                                        float result =
                                                vectorDot(
                                                        bQ8_0Span,
                                                        bOffset + bRowStride * c,
                                                        aArraySpan,
                                                        (int) (aOffset + aRowStride * r),
                                                        (int) K);
                                        outAccess.setElementAt(
                                                out, outOffset + outRowStride * r + c, result);
                                    });
                        });
                return;
            }

            if (b instanceof Q4_1Span bQ4_1Span) {
                Parallel.parallelForLong(
                        0,
                        R,
                        r -> {
                            Parallel.parallelForLong(
                                    0,
                                    C,
                                    c -> {
                                        float result =
                                                vectorDot(
                                                        bQ4_1Span,
                                                        bOffset + bRowStride * c,
                                                        aArraySpan,
                                                        (int) (aOffset + aRowStride * r),
                                                        (int) K);
                                        outAccess.setElementAt(
                                                out, outOffset + outRowStride * r + c, result);
                                    });
                        });
                return;
            }

            if (b instanceof BF16Span bBF16Span) {
                Parallel.parallelForLong(
                        0,
                        R,
                        r -> {
                            Parallel.parallelForLong(
                                    0,
                                    C,
                                    c -> {
                                        float result =
                                                vectorDot(
                                                        bBF16Span,
                                                        bOffset + bRowStride * c,
                                                        aArraySpan,
                                                        (int) (aOffset + aRowStride * r),
                                                        (int) K);
                                        outAccess.setElementAt(
                                                out, outOffset + outRowStride * r + c, result);
                                    });
                        });
                return;
            }

            if (b instanceof ArraySpan bArraySpan) {
                // ArraySpan outArraySpan = (ArraySpan) out;

                //                BLAS.sgemm(
                ////                //Sgemm.sgemm(
                //////                BLAS.getInstance().sgemm(
                //                        "T", "N",
                //                        (int) C, (int) R, (int) K,
                //                        1f,
                //                        bArraySpan.values, bArraySpan.offset + (int) bOffset,
                // (int) bRowStride,
                //                        aArraySpan.values, aArraySpan.offset + (int) aOffset,
                // (int) aRowStride,
                //                        0f,
                //                        outArraySpan.values, outArraySpan.offset + (int)
                // outOffset, (int) outRowStride
                //                );

                Parallel.parallelForLong(
                        0,
                        R,
                        r -> {
                            // for (int r = 0; r < R; ++r) {
                            Parallel.parallelForLong(
                                    0,
                                    C,
                                    c -> {
                                        // for (int c = 0; c < C; ++c) {
                                        float result =
                                                vectorDot(
                                                        bArraySpan,
                                                        bOffset + bRowStride * c,
                                                        aArraySpan,
                                                        (int) (aOffset + aRowStride * r),
                                                        (int) K);
                                        outAccess.setElementAt(
                                                out, outOffset + outRowStride * r + c, result);
                                    });
                        });
                return;
            }
        }

        super.gemmRowMajor(
                R,
                C,
                K,
                a,
                aOffset,
                aRowStride,
                b,
                bOffset,
                bRowStride,
                out,
                outOffset,
                outRowStride);
    }

    @Override
    public void fill(float value, FloatSpan out) {
        if (out instanceof ArraySpan arraySpan) {
            FloatVector vectorValue = FloatVector.broadcast(F_SPECIES, value);
            long upperBound = F_SPECIES.loopBound(arraySpan.size());
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                vectorValue.intoArray(arraySpan.values, arraySpan.offset + i);
            }
            // tail
            if (i < out.size()) {
                var spanAccess = directAccess.apply(out);
                for (; i < out.size(); ++i) {
                    spanAccess.setElementAt(out, i, value);
                }
            }
        } else {
            super.fill(value, out);
        }
    }

    @Override
    public void scale(FloatSpan in, float value, FloatSpan out) {
        // In-place mutation (in == out) must be supported, arbitrary overlap is not supported.
        assert out.size() >= in.size();
        if (in instanceof ArraySpan inArraySpan && out instanceof ArraySpan outArraySpan) {
            long upperBound = F_SPECIES.loopBound(inArraySpan.size());
            int i = 0;
            for (; i < upperBound; i += F_SPECIES.length()) {
                FloatVector vectorValue =
                        FloatVector.fromArray(
                                F_SPECIES, inArraySpan.values, inArraySpan.offset + i);
                vectorValue = vectorValue.mul(value);
                vectorValue.intoArray(outArraySpan.values, outArraySpan.offset + i);
            }
            // tail
            if (i < inArraySpan.size()) {
                var spanAccess = directAccess.apply(in);
                var outAccess = directAccess.apply(out);
                for (; i < inArraySpan.size(); ++i) {
                    outAccess.setElementAt(out, i, spanAccess.getElementAt(out, i) * value);
                }
            }
        } else {
            super.scale(in, value, out);
        }
    }

    private float vectorDot(
            Q8_0Span thiz, long thisOffset, ArraySpan that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + startIndex to type().getElementsPerBlock()().
        assert Integer.bitCount(GGMLType.Q8_0.getElementsPerBlock()) == 1 : "power of 2";
        long alignmentBound =
                Math.min(size, -thisOffset & (GGMLType.Q8_0.getElementsPerBlock() - 1));
        if (alignmentBound > 0) {
            result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q8_0.getElementsPerBlock() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        long blockOffset =
                (thisOffset + j)
                        / GGMLType.Q8_0.getElementsPerBlock()
                        * GGMLType.Q8_0.getBlockByteSize();
        int upperBound =
                size / GGMLType.Q8_0.getElementsPerBlock() * GGMLType.Q8_0.getElementsPerBlock();
        for (;
                j < upperBound;
                j += GGMLType.Q8_0.getElementsPerBlock(),
                        blockOffset += GGMLType.Q8_0.getBlockByteSize()) {
            float wScaleValue =
                    Float.float16ToFloat(Util.readShort(thiz.memorySegment, blockOffset));
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var wBytes =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_256,
                                    thiz.memorySegment,
                                    blockOffset + Float16.BYTES,
                                    ByteOrder.LITTLE_ENDIAN);
                    var sum0 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 0 * F_SPECIES.length())
                                    .mul(wBytes.castShape(F_SPECIES, 0));
                    var sum1 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 1 * F_SPECIES.length())
                                    .mul(wBytes.castShape(F_SPECIES, 1));
                    val = sum0.add(sum1).fma(wScale, val);
                }
                case 256 -> {
                    var wBytes =
                            ByteVector.fromMemorySegment(
                                    ByteVector.SPECIES_256,
                                    thiz.memorySegment,
                                    blockOffset + Float16.BYTES,
                                    ByteOrder.LITTLE_ENDIAN);
                    var sum0 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 0 * F_SPECIES.length())
                                    .mul(wBytes.castShape(F_SPECIES, 0));
                    var sum1 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 1 * F_SPECIES.length())
                                    .mul(wBytes.castShape(F_SPECIES, 1));
                    var sum2 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 2 * F_SPECIES.length())
                                    .mul(wBytes.castShape(F_SPECIES, 2));
                    var sum3 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 3 * F_SPECIES.length())
                                    .mul(wBytes.castShape(F_SPECIES, 3));
                    val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                }
                case 128 -> {
                    VectorSpecies<Byte> B_128 = ByteVector.SPECIES_128;
                    // This loop cannot be unrolled, why?
                    for (int i = 0; i < 2; ++i) {
                        var wBytes =
                                ByteVector.fromMemorySegment(
                                        B_128,
                                        thiz.memorySegment,
                                        blockOffset + Float16.BYTES + i * B_128.vectorByteSize(),
                                        ByteOrder.LITTLE_ENDIAN);
                        var sum0 =
                                getFloatVector(
                                                F_SPECIES,
                                                that,
                                                thatOffset + j + i * 16 + 0 * F_SPECIES.length())
                                        .mul(wBytes.castShape(F_SPECIES, 0));
                        var sum1 =
                                getFloatVector(
                                                F_SPECIES,
                                                that,
                                                thatOffset + j + i * 16 + 1 * F_SPECIES.length())
                                        .mul(wBytes.castShape(F_SPECIES, 1));
                        var sum2 =
                                getFloatVector(
                                                F_SPECIES,
                                                that,
                                                thatOffset + j + i * 16 + 2 * F_SPECIES.length())
                                        .mul(wBytes.castShape(F_SPECIES, 2));
                        var sum3 =
                                getFloatVector(
                                                F_SPECIES,
                                                that,
                                                thatOffset + j + i * 16 + 3 * F_SPECIES.length())
                                        .mul(wBytes.castShape(F_SPECIES, 3));
                        val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        // Remaining entries.
        if (j < size) {
            result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

    private float vectorDot(
            BF16Span thiz, long thisOffset, ArraySpan that, int thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        long upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = getFloatVector(F_SPECIES, that, thatOffset + i);
            ShortVector bfloat16 =
                    ShortVector.fromMemorySegment(
                            S_SPECIES_HALF,
                            thiz.memorySegment,
                            (thisOffset + i) * (long) BFloat16.BYTES,
                            ByteOrder.LITTLE_ENDIAN);
            FloatVector thizVector =
                    bfloat16.castShape(I_SPECIES, 0) // (int) vi
                            .lanewise(VectorOperators.LSHL, 16) // vi <<= 16
                            .reinterpretAsFloats(); // Float.intBitsToFloat(vi)
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);

        // Remaining entries.
        if (upperBound < size) {
            result +=
                    scalarDot(
                            thiz,
                            thisOffset + upperBound,
                            that,
                            thatOffset + upperBound,
                            size - upperBound);
        }

        return result;
    }

    private float vectorDot(
            ArraySpan thiz, long thisOffset, ArraySpan that, int thatOffset, int size) {
        assert S_SPECIES_HALF.length() == F_SPECIES.length();
        FloatVector val = FloatVector.zero(F_SPECIES);
        long upperBound = F_SPECIES.loopBound(size);
        for (int i = 0; i < upperBound; i += F_SPECIES.length()) {
            FloatVector thatVector = getFloatVector(F_SPECIES, that, thatOffset + i);
            FloatVector thizVector =
                    FloatVector.fromArray(
                            F_SPECIES, thiz.values, thiz.offset + (int) thisOffset + i);
            val = thizVector.fma(thatVector, val);
        }
        float result = val.reduceLanes(VectorOperators.ADD);

        // Remaining entries.
        if (upperBound < size) {
            result +=
                    scalarDot(
                            thiz,
                            thisOffset + upperBound,
                            that,
                            thatOffset + upperBound,
                            size - upperBound);
        }

        return result;
    }

    private float vectorDot(
            Q4_0Span thiz, long thisOffset, ArraySpan that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + j to type().getElementsPerBlock().
        assert Integer.bitCount(GGMLType.Q4_0.getElementsPerBlock()) == 1 : "power of 2";
        long alignmentBound =
                Math.min(size, -thisOffset & (GGMLType.Q4_0.getElementsPerBlock() - 1));
        if (alignmentBound > 0) {
            result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q4_0.getElementsPerBlock() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        long blockOffset =
                (thisOffset + j)
                        / GGMLType.Q4_0.getElementsPerBlock()
                        * GGMLType.Q4_0.getBlockByteSize();
        int upperBound =
                size / GGMLType.Q4_0.getElementsPerBlock() * GGMLType.Q4_0.getElementsPerBlock();
        for (;
                j < upperBound;
                j += GGMLType.Q4_0.getElementsPerBlock(),
                        blockOffset += GGMLType.Q4_0.getBlockByteSize()) {
            float wScaleValue =
                    Float.float16ToFloat(Util.readShort(thiz.memorySegment, blockOffset));
            var B_SPECIES = ByteVector.SPECIES_128;
            var wBytes =
                    ByteVector.fromMemorySegment(
                            B_SPECIES,
                            thiz.memorySegment,
                            blockOffset + Float16.BYTES,
                            ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF).sub((byte) 8);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4).sub((byte) 8);
            var wScale = FloatVector.broadcast(F_SPECIES, wScaleValue);
            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var sum0 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 0 * F_SPECIES.length())
                                    .mul(loBytes.castShape(F_SPECIES, 0));
                    var sum1 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 1 * F_SPECIES.length())
                                    .mul(hiBytes.castShape(F_SPECIES, 0));
                    val = sum0.add(sum1).fma(wScale, val);
                }
                case 256 -> {
                    var sum0 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 0 * F_SPECIES.length())
                                    .mul(loBytes.castShape(F_SPECIES, 0));
                    var sum1 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 1 * F_SPECIES.length())
                                    .mul(loBytes.castShape(F_SPECIES, 1));
                    var sum2 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 2 * F_SPECIES.length())
                                    .mul(hiBytes.castShape(F_SPECIES, 0));
                    var sum3 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 3 * F_SPECIES.length())
                                    .mul(hiBytes.castShape(F_SPECIES, 1));
                    val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                }
                case 128 -> {
                    // This loop cannot be unrolled, why?
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var sum0 =
                                getFloatVector(
                                                F_SPECIES,
                                                that,
                                                thatOffset + j + (i * 4 + 0) * F_SPECIES.length())
                                        .mul(tmp.castShape(F_SPECIES, 0));
                        var sum1 =
                                getFloatVector(
                                                F_SPECIES,
                                                that,
                                                thatOffset + j + (i * 4 + 1) * F_SPECIES.length())
                                        .mul(tmp.castShape(F_SPECIES, 1));
                        var sum2 =
                                getFloatVector(
                                                F_SPECIES,
                                                that,
                                                thatOffset + j + (i * 4 + 2) * F_SPECIES.length())
                                        .mul(tmp.castShape(F_SPECIES, 2));
                        var sum3 =
                                getFloatVector(
                                                F_SPECIES,
                                                that,
                                                thatOffset + j + (i * 4 + 3) * F_SPECIES.length())
                                        .mul(tmp.castShape(F_SPECIES, 3));
                        val = sum0.add(sum1).add(sum2).add(sum3).fma(wScale, val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        // Remaining entries.
        if (j < size) {
            result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

    private float vectorDot(
            Q4_1Span thiz, long thisOffset, ArraySpan that, int thatOffset, int size) {
        float result = 0f;
        int j = 0;

        // Align thisOffset + j to type().getElementsPerBlock().
        assert Integer.bitCount(GGMLType.Q4_1.getElementsPerBlock()) == 1 : "power of 2";
        long alignmentBound =
                Math.min(size, -thisOffset & (GGMLType.Q4_1.getElementsPerBlock() - 1));
        if (alignmentBound > 0) {
            result += scalarDot(thiz, thisOffset, that, thatOffset, alignmentBound);
            j += alignmentBound;
        }
        assert (thisOffset + j) % GGMLType.Q4_1.getElementsPerBlock() == 0;

        FloatVector val = FloatVector.zero(F_SPECIES);
        long blockOffset =
                (thisOffset + j)
                        / GGMLType.Q4_1.getElementsPerBlock()
                        * GGMLType.Q4_1.getBlockByteSize();
        int upperBound =
                size / GGMLType.Q4_1.getElementsPerBlock() * GGMLType.Q4_1.getElementsPerBlock();
        for (;
                j < upperBound;
                j += GGMLType.Q4_1.getElementsPerBlock(),
                        blockOffset += GGMLType.Q4_1.getBlockByteSize()) {
            float scale = Float.float16ToFloat(Util.readShort(thiz.memorySegment, blockOffset));
            float offset =
                    Float.float16ToFloat(
                            Util.readShort(thiz.memorySegment, blockOffset + Float16.BYTES));

            var B_SPECIES = ByteVector.SPECIES_128;
            var wBytes =
                    ByteVector.fromMemorySegment(
                            B_SPECIES,
                            thiz.memorySegment,
                            blockOffset + Float16.BYTES + Float16.BYTES,
                            ByteOrder.LITTLE_ENDIAN);
            var loBytes = wBytes.and((byte) 0xF);
            var hiBytes = wBytes.lanewise(VectorOperators.LSHR, 4);

            switch (F_SPECIES.vectorBitSize()) {
                case 512 -> {
                    var sum0 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 0 * F_SPECIES.length())
                                    .mul(
                                            loBytes.castShape(F_SPECIES, 0)
                                                    .reinterpretAsFloats()
                                                    .fma(scale, offset));
                    var sum1 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 1 * F_SPECIES.length())
                                    .mul(
                                            hiBytes.castShape(F_SPECIES, 0)
                                                    .reinterpretAsFloats()
                                                    .fma(scale, offset));
                    val = sum0.add(sum1).add(val);
                }
                case 256 -> {
                    var sum0 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 0 * F_SPECIES.length())
                                    .mul(
                                            loBytes.castShape(F_SPECIES, 0)
                                                    .reinterpretAsFloats()
                                                    .fma(scale, offset));
                    var sum1 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 1 * F_SPECIES.length())
                                    .mul(
                                            loBytes.castShape(F_SPECIES, 1)
                                                    .reinterpretAsFloats()
                                                    .fma(scale, offset));
                    var sum2 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 2 * F_SPECIES.length())
                                    .mul(
                                            hiBytes.castShape(F_SPECIES, 0)
                                                    .reinterpretAsFloats()
                                                    .fma(scale, offset));
                    var sum3 =
                            getFloatVector(F_SPECIES, that, thatOffset + j + 3 * F_SPECIES.length())
                                    .mul(
                                            hiBytes.castShape(F_SPECIES, 1)
                                                    .reinterpretAsFloats()
                                                    .fma(scale, offset));
                    val = sum0.add(sum1).add(sum2).add(sum3).add(val);
                }
                case 128 -> {
                    // This loop cannot be unrolled, why?
                    for (int i = 0; i < 2; ++i) {
                        var tmp = i == 0 ? loBytes : hiBytes;
                        var sum0 =
                                getFloatVector(
                                                F_SPECIES,
                                                that,
                                                thatOffset + j + (i * 4 + 0) * F_SPECIES.length())
                                        .mul(
                                                tmp.castShape(F_SPECIES, 0)
                                                        .reinterpretAsFloats()
                                                        .fma(scale, offset));
                        var sum1 =
                                getFloatVector(
                                                F_SPECIES,
                                                that,
                                                thatOffset + j + (i * 4 + 1) * F_SPECIES.length())
                                        .mul(
                                                tmp.castShape(F_SPECIES, 1)
                                                        .reinterpretAsFloats()
                                                        .fma(scale, offset));
                        var sum2 =
                                getFloatVector(
                                                F_SPECIES,
                                                that,
                                                thatOffset + j + (i * 4 + 2) * F_SPECIES.length())
                                        .mul(
                                                tmp.castShape(F_SPECIES, 2)
                                                        .reinterpretAsFloats()
                                                        .fma(scale, offset));
                        var sum3 =
                                getFloatVector(
                                                F_SPECIES,
                                                that,
                                                thatOffset + j + (i * 4 + 3) * F_SPECIES.length())
                                        .mul(
                                                tmp.castShape(F_SPECIES, 3)
                                                        .reinterpretAsFloats()
                                                        .fma(scale, offset));
                        val = sum0.add(sum1).add(sum2).add(sum3).add(val);
                    }
                }
                default -> throw new UnsupportedOperationException(F_SPECIES.toString());
            }
        }
        result += val.reduceLanes(VectorOperators.ADD);

        // Remaining entries.
        if (j < size) {
            result += scalarDot(thiz, thisOffset + j, that, thatOffset + j, size - j);
        }

        return result;
    }

    private static FloatVector getFloatVector(
            VectorSpecies<Float> species, ArraySpan span, int index) {
        return FloatVector.fromArray(species, span.values, span.offset + index);
    }

    //    private static FloatVector getFloatVectorInt(VectorSpecies<Float> species, ArraySpan span,
    // int index) {
    //        return FloatVector.fromArray(species, span.values, span.offset + index);
    //    }
    //
    //    private static FloatVector getFloatVector(VectorSpecies<Float> species, F32Span span, long
    // index) {
    //        long offsetInBytes = index * (long) Float.BYTES;
    //        return FloatVector.fromMemorySegment(species, span.memorySegment, offsetInBytes,
    // ByteOrder.nativeOrder());
    //    }
}
