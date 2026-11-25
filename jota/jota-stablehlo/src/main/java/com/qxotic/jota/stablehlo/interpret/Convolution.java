package com.qxotic.jota.stablehlo.interpret;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.ConvDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.IntegerAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.type.FloatFormat;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.TensorType;
import java.util.Arrays;

/**
 * {@code convolution} and {@code dynamic_conv} ({@code docs/spec.md}): every output element sums
 * the products of an input window (strided, padded, lhs- and rhs-dilated, optionally reversed
 * kernel) with the kernel, within its feature group and batch group. Sums are accumulated like
 * {@code dot_general}: in f32 for f32 and narrower results, double for f64, long for integers.
 */
final class Convolution {

    private Convolution() {}

    /** {@code padding} is the operand of {@code dynamic_conv}, null for {@code convolution}. */
    static Tensor convolve(Operation op, Tensor lhs, Tensor rhs, Tensor padding) {
        if (lhs.isComplex() || rhs.isComplex()) {
            throw new InterpreterException("convolution of complex values");
        }
        ConvDimensionNumbers dn =
                ConvDimensionNumbers.from(Interpreter.attr(op, "dimension_numbers"));
        TensorType rt = Interpreter.resultType(op, 0);
        int s = dn.inputSpatialDimensions().length;
        long[] strides = ones(Interpreter.longs(op, "window_strides"), s);
        long[] lhsDil = ones(Interpreter.longs(op, "lhs_dilation"), s);
        long[] rhsDil = ones(Interpreter.longs(op, "rhs_dilation"), s);
        long[] padLow = new long[s];
        long[] pads = padding != null ? padValues(padding) : Interpreter.longs(op, "padding");
        if (pads != null) {
            for (int d = 0; d < s; d++) {
                padLow[d] = pads[2 * d];
            }
        }
        boolean[] reverse = new boolean[s];
        Attribute rev = op.attribute("window_reversal").orElse(null);
        if (rev instanceof DenseArrayAttr r && r.size() == s) {
            reverse = r.booleans();
        }
        long fgc = longOr(op, "feature_group_count", 1);
        long bgc = longOr(op, "batch_group_count", 1);

        long[] ld = lhs.dims();
        long[] kd = rhs.dims();
        long[] od = rt.dims();
        long kIn = kd[(int) dn.kernelInputFeatureDimension()];
        long kOut = kd[(int) dn.kernelOutputFeatureDimension()];
        long outBatch = od[(int) dn.outputBatchDimension()];
        long[] window = new long[s];
        for (int d = 0; d < s; d++) {
            window[d] = kd[(int) dn.kernelSpatialDimensions()[d]];
        }
        int windowSize = (int) Arrays.stream(window).reduce(1, (a, b) -> a * b);

        boolean floatResult = rt.elementType() instanceof FloatType;
        boolean f32 = floatResult && ((FloatType) rt.elementType()).format() != FloatFormat.F64;
        int n = (int) rt.numElements();
        double[] fo = floatResult ? new double[n] : null;
        long[] io = floatResult ? null : new long[n];
        long[] li = new long[ld.length];
        long[] ki = new long[kd.length];
        for (int i = 0; i < n; i++) {
            long[] oi = Tensor.unravel(i, od);
            long b = oi[(int) dn.outputBatchDimension()];
            long f = oi[(int) dn.outputFeatureDimension()];
            long featureGroup = f / (kOut / fgc);
            long batchGroup = f / (kOut / bgc);
            double fs = 0;
            long is = 0;
            for (long c = 0; c < kIn; c++) {
                li[(int) dn.inputBatchDimension()] = batchGroup * outBatch + b;
                li[(int) dn.inputFeatureDimension()] = featureGroup * kIn + c;
                ki[(int) dn.kernelInputFeatureDimension()] = c;
                ki[(int) dn.kernelOutputFeatureDimension()] = f;
                for (int w = 0; w < windowSize; w++) {
                    long[] wi = Tensor.unravel(w, window);
                    boolean inside = true;
                    for (int d = 0; d < s && inside; d++) {
                        long o = oi[(int) dn.outputSpatialDimensions()[d]];
                        long p = o * strides[d] + wi[d] * rhsDil[d] - padLow[d];
                        long size = ld[(int) dn.inputSpatialDimensions()[d]];
                        if (p < 0 || p % lhsDil[d] != 0 || p / lhsDil[d] >= size) {
                            inside = false;
                        } else {
                            li[(int) dn.inputSpatialDimensions()[d]] = p / lhsDil[d];
                            ki[(int) dn.kernelSpatialDimensions()[d]] =
                                    reverse[d] ? window[d] - 1 - wi[d] : wi[d];
                        }
                    }
                    if (!inside) {
                        continue;
                    }
                    int a = (int) Tensor.ravel(li, ld);
                    int k = (int) Tensor.ravel(ki, kd);
                    if (floatResult) {
                        double prod = lhs.getDouble(a) * rhs.getDouble(k);
                        fs = Interpreter.acc(fs, f32 ? (float) prod : prod, f32);
                    } else {
                        is += lhs.getLong(a) * rhs.getLong(k);
                    }
                }
            }
            if (floatResult) {
                fo[i] = fs;
            } else {
                io[i] = is;
            }
        }
        if (floatResult) {
            return Tensor.ofDoubles(rt, fo);
        }
        if (((IntegerType) rt.elementType()).isBoolean()) {
            for (int i = 0; i < n; i++) {
                io[i] = io[i] != 0 ? 1 : 0;
            }
        }
        return Tensor.ofLongs(rt, io);
    }

    private static long[] padValues(Tensor padding) {
        long[] v = new long[padding.size()];
        for (int i = 0; i < v.length; i++) {
            v[i] = padding.getLong(i);
        }
        return v;
    }

    private static long[] ones(long[] v, int n) {
        if (v != null && v.length == n) {
            return v;
        }
        long[] r = new long[n];
        Arrays.fill(r, 1);
        return r;
    }

    private static long longOr(Operation op, String name, long fallback) {
        return op.attribute(name).orElse(null) instanceof IntegerAttr i ? i.value() : fallback;
    }
}
