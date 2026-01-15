package ai.qxotic.model.llm.llama;

import ai.qxotic.span.*;

import java.util.function.Function;

public class BaseKernelOps implements KernelOps<FloatSpan, FloatMatrixView> {

    final Function<FloatSpan, DirectAccessOps<FloatSpan>> directAccess;

    public BaseKernelOps(Function<FloatSpan, DirectAccessOps<FloatSpan>> directAccess) {
        this.directAccess = directAccess;
    }

    @Override
    public void elementWise(FloatSpan span, FloatUnaryOperator mapper, FloatSpan out) {
        var spanAccess = directAccess.apply(span);
        var outAccess = directAccess.apply(out);
        for (long i = 0; i < span.size(); ++i) {
            outAccess.setElementAt(out, i, mapper.apply(spanAccess.getElementAt(span, i)));
        }
    }

    @Override
    public void map2(FloatSpan span, FloatSpan other, FloatBinaryOperator combiner, FloatSpan out) {
        var spanAccess = directAccess.apply(span);
        var otherAccess = directAccess.apply(other);
        var outAccess = directAccess.apply(out);
        for (long i = 0; i < span.size(); ++i) {
            outAccess.setElementAt(out, i, combiner.apply(spanAccess.getElementAt(span, i), otherAccess.getElementAt(other, i)));
        }
    }

    @Override
    public void copyTo(FloatSpan span, FloatSpan out) {
        var spanAccess = directAccess.apply(span);
        var outAccess = directAccess.apply(out);
        for (long i = 0; i < span.size(); ++i) {
            outAccess.setElementAt(out, i, spanAccess.getElementAt(span, i));
        }
    }

    @Override
    public void copyToStrided(FloatSpan in, FloatSpan out, long outOffset, long outElementStride) {
        var inAccess = directAccess.apply(in);
        var outAccess = directAccess.apply(out);
        for (long i = 0; i < in.size(); ++i) {
            outAccess.setElementAt(out, outOffset + i * outElementStride, inAccess.getElementAt(in, i));
        }
    }

    @Override
    public void softMax(FloatSpan span, FloatSpan out) {
        DirectAccessOps<FloatSpan> spanAccess = directAccess.apply(span);
        DirectAccessOps<FloatSpan> outAccess = directAccess.apply(out);
        // find max value (for numerical stability)
        float maxValue = spanAccess.max(span);
        // exp and sum
        elementWise(span, f -> (float) Math.exp(f - maxValue), out);
        float sum = outAccess.sum(out);
        // normalize
        scale(out, 1f / sum, out);
    }

    @Override
    public void rmsNorm(FloatSpan span, FloatSpan weight, float rmsNormEps, FloatSpan out) {
        // calculate sum of squares
        DirectAccessOps<FloatSpan> spanAccess = directAccess.apply(span);
        float ss = spanAccess.fold(span, 0f, FloatBinaryOperator.SUM_OF_SQUARES);
        ss /= span.size();
        ss += rmsNormEps;
        ss = (float) (1.0 / Math.sqrt(ss));
        // normalize and scale
        final float finalss = ss;
        map2(span, weight, (f, s) -> finalss * f * s, out);
    }

    @Override
    public void matrixVectorMultiply(FloatMatrixView matrix, FloatSpan vector, FloatSpan out) {
        var outAccess = directAccess.apply(out);
        long rows = matrix.rows();
        long cols = matrix.cols();
        assert out.size() == rows;
        assert vector.size() == cols;
        Parallel.parallelForLong(0, rows, r -> {
            float result = scalarDot(matrix.innerSpan(), matrix.rowOffset(r), vector, 0, cols);
            outAccess.setElementAt(out, r, result);
        });
    }

//    // A @ B = C
//    // [rows, cols] @ [cols, K]^T = [rows, K]^T
//    @Override
//    public void matrixMultiply(FloatMatrixView a, int batchSize, FloatMatrixView b, FloatMatrixView out) {
//        long rows = a.rows();
//        long cols = a.cols();
//        var outAccess = directAccess.apply(out.innerSpan());
//        for (int r = 0; r < rows; ++r) {
//            for (int k = 0; k < batchSize; ++k) {
//                float result = scalarDot(a.innerSpan(), a.rowOffset(r), b.innerSpan(), b.rowOffset(k), cols);
//                outAccess.setElementAt(out.innerSpan(), out.rowOffset(k) + r, result);
//            }
//        }
//    }

    @Override
    public void gemmRowMajor(long R, long C, long K,
                             FloatSpan a, long aOffset, long aRowStride, // [R, K]
                             FloatSpan b, long bOffset, long bRowStride, // [K, C]^T
                             FloatSpan out, long outOffset, long outRowStride) { // [R, C]
        var outAccess = directAccess.apply(out);
        Parallel.parallelForLong(0, R, r -> {
            //for (int r = 0; r < R; ++r) {
            Parallel.parallelForLong(0, C, c -> {
                //for (int c = 0; c < C; ++c) {
                float result = scalarDot(a, aOffset + aRowStride * r, b, bOffset + bRowStride * c, K);
                outAccess.setElementAt(out, outOffset + outRowStride * r + c, result);
            });
        });
    }

    @Override
    public void rotate(boolean neoxStyle, FloatSpan span, FloatSpan freqReal, FloatSpan freqImag, int position, int numberOfHeads, int headSize, FloatSpan out) {
        assert span.size() == numberOfHeads * (long) headSize;

        var freqAccess = directAccess.apply(freqReal);
        var spanAccess = directAccess.apply(span);
        var outAccess = directAccess.apply(out);

        if (neoxStyle) {
            // GPT-NeoX style RoPE, real/imaginary components are stored with a headSize/2 offset per head, instead of consecutive.
            for (int h = 0; h < numberOfHeads; ++h) {
                int poffset = h * headSize;
                for (int i0 = 0; i0 < headSize; i0 += 2) {
                    int ic = i0 / 2;
                    float fcr = freqAccess.getElementAt(freqReal, position * (headSize / 2) + ic);
                    float fci = freqAccess.getElementAt(freqImag, position * (headSize / 2) + ic);
                    float v0 = spanAccess.getElementAt(span, poffset + ic);
                    float v1 = spanAccess.getElementAt(span, poffset + ic + headSize / 2);
                    outAccess.setElementAt(out, poffset + ic, v0 * fcr - v1 * fci);
                    outAccess.setElementAt(out, poffset + ic + headSize / 2, v0 * fci + v1 * fcr);
                }
            }
        } else {
            // Traditional layout, real/imaginary components are stored consecutive.
            for (int i = 0; i < span.size(); i += 2) {
                int head_dim = i % headSize;
                float fcr = freqAccess.getElementAt(freqReal, position * (headSize / 2) + (head_dim / 2));
                float fci = freqAccess.getElementAt(freqImag, position * (headSize / 2) + (head_dim / 2));
                float v0 = spanAccess.getElementAt(span, i);
                float v1 = spanAccess.getElementAt(span, i + 1);
                outAccess.setElementAt(out, i, v0 * fcr - v1 * fci);
                outAccess.setElementAt(out, i + 1, v0 * fci + v1 * fcr);
            }
        }
    }

    @Override
    public void fill(float value, FloatSpan out) {
        var spanAccess = directAccess.apply(out);
        for (long i = 0; i < out.size(); ++i) {
            spanAccess.setElementAt(out, i, value);
        }
    }

    @Override
    public void scale(FloatSpan span, float value, FloatSpan out) {
        elementWise(span, f -> f * value, out);
    }



    public void transposeInPlace(
            long R, int C, FloatSpan matrix, long offset, long rowStride) {
        var access = directAccess.apply(matrix);

        for (long r = 0; r < R; r++) {
            for (long c = r + 1; c < C; c++) {
                // Swap elements [r,c] and [c,r]
                long pos1 = offset + r * rowStride + c;
                long pos2 = offset + c * rowStride + r;

                float temp = access.getElementAt(matrix, pos1);
                access.setElementAt(matrix, pos1, access.getElementAt(matrix, pos2));
                access.setElementAt(matrix, pos2, temp);
            }
        }
    }

    float scalarDot(FloatSpan thiz, long thizOffset, FloatSpan that, long thatOffset, long size) {
        var thizAccess = directAccess.apply(thiz);
        var thatAccess = directAccess.apply(that);
        float result = 0f;
        for (long i = 0; i < size; ++i) {
            result += thizAccess.getElementAt(thiz, thizOffset + i) * thatAccess.getElementAt(that, thatOffset + i);
        }
        return result;
    }
}
