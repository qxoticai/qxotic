package com.qxotic.jota.stablehlo.type;

import java.util.List;
import java.util.Objects;

/**
 * Uniform quantized element type {@code !quant.uniform<storage<min:max>:expressed[:dim], params>}
 * (spec "QuantizedTensorElementType", constraints C1-C11 are checked by the verifier).
 *
 * @param storageMin explicit storage minimum, or null for {@code min_value(storage_type)}
 * @param storageMax explicit storage maximum, or null for {@code max_value(storage_type)}
 * @param quantizationDimension per-axis dimension, or null for per-tensor quantization
 */
public record QuantizedType(
        IntegerType storageType,
        Long storageMin,
        Long storageMax,
        FloatType expressedType,
        Integer quantizationDimension,
        List<Double> scales,
        List<Long> zeroPoints)
        implements ElementType {

    public QuantizedType {
        Objects.requireNonNull(storageType, "storageType");
        Objects.requireNonNull(expressedType, "expressedType");
        scales = List.copyOf(scales);
        zeroPoints = List.copyOf(zeroPoints);
        if (scales.size() != zeroPoints.size()) {
            throw new IllegalArgumentException("scales and zero points differ in size");
        }
        // a range equal to the storage type's full range is the default: keep one spelling
        int bits = storageType.bitWidth();
        long defaultMin = storageType.isUnsigned() ? 0 : -(1L << (bits - 1));
        long defaultMax = storageType.isUnsigned() ? (1L << bits) - 1 : (1L << (bits - 1)) - 1;
        if ((storageMin == null || storageMin == defaultMin)
                && (storageMax == null || storageMax == defaultMax)) {
            storageMin = null;
            storageMax = null;
        }
    }

    public boolean isPerAxis() {
        return quantizationDimension != null;
    }

    @Override
    public int storageBytes() {
        return storageType.storageBytes();
    }

    @Override
    public String mlir() {
        StringBuilder sb = new StringBuilder("!quant.uniform<").append(storageType.mlir());
        if (storageMin != null || storageMax != null) {
            sb.append('<').append(storageMin).append(':').append(storageMax).append('>');
        }
        sb.append(':').append(expressedType.mlir());
        if (quantizationDimension != null) {
            sb.append(':').append(quantizationDimension);
        }
        sb.append(", ");
        if (quantizationDimension != null) {
            sb.append('{');
        }
        for (int i = 0; i < scales.size(); i++) {
            if (i > 0) {
                sb.append(',');
            }
            sb.append(formatScale(scales.get(i)));
            if (zeroPoints.get(i) != 0) {
                sb.append(':').append(zeroPoints.get(i));
            }
        }
        if (quantizationDimension != null) {
            sb.append('}');
        }
        return sb.append('>').toString();
    }

    private static String formatScale(double scale) {
        return String.format(java.util.Locale.ROOT, "%.17e", scale);
    }

    @Override
    public String toString() {
        return mlir();
    }
}
