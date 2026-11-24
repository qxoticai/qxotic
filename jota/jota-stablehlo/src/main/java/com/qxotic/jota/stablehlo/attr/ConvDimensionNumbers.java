package com.qxotic.jota.stablehlo.attr;

import java.util.Arrays;

/**
 * {@code #stablehlo.conv<...>} (spec "convolution", inputs I8-I16).
 *
 * <p>Parses both spellings of the attribute body: the compact layout form {@code [b, 0, 1, f]x[0,
 * 1, i, o]->[b, 0, 1, f]} and the {@code raw input_batch_dimension = 0, ...} form.
 */
public record ConvDimensionNumbers(
        long inputBatchDimension,
        long inputFeatureDimension,
        long[] inputSpatialDimensions,
        long kernelInputFeatureDimension,
        long kernelOutputFeatureDimension,
        long[] kernelSpatialDimensions,
        long outputBatchDimension,
        long outputFeatureDimension,
        long[] outputSpatialDimensions) {

    public static ConvDimensionNumbers from(Attribute attribute) {
        if (attribute instanceof DialectAttr d
                && d.dialect().equals("stablehlo")
                && d.mnemonic().equals("conv")) {
            if (d.hasParams()) {
                return fromRaw(d);
            }
            String body = d.rawBody().trim();
            if (body.startsWith("raw")) {
                return fromRaw(DialectAttr.ofParams("stablehlo", "conv", parseRaw(body)));
            }
            return parseCompact(body);
        }
        throw new IllegalArgumentException("expected #stablehlo.conv<...>, got " + attribute);
    }

    private static ConvDimensionNumbers fromRaw(DialectAttr d) {
        return new ConvDimensionNumbers(
                DimensionNumbers.scalar(d, "input_batch_dimension", 0),
                DimensionNumbers.scalar(d, "input_feature_dimension", 0),
                DimensionNumbers.longs(d, "input_spatial_dimensions"),
                DimensionNumbers.scalar(d, "kernel_input_feature_dimension", 0),
                DimensionNumbers.scalar(d, "kernel_output_feature_dimension", 0),
                DimensionNumbers.longs(d, "kernel_spatial_dimensions"),
                DimensionNumbers.scalar(d, "output_batch_dimension", 0),
                DimensionNumbers.scalar(d, "output_feature_dimension", 0),
                DimensionNumbers.longs(d, "output_spatial_dimensions"));
    }

    /** Parses {@code raw key = 0, key = [1, 2], ...} into i64 / i64-array parameters. */
    private static java.util.List<NamedAttribute> parseRaw(String body) {
        java.util.List<NamedAttribute> params = new java.util.ArrayList<>();
        java.util.regex.Matcher m =
                java.util.regex.Pattern.compile("(\\w+)\\s*=\\s*(\\[[^\\]]*\\]|-?\\d+)")
                        .matcher(body.substring(3));
        while (m.find()) {
            String value = m.group(2);
            if (value.startsWith("[")) {
                String inner = value.substring(1, value.length() - 1).trim();
                long[] values =
                        inner.isEmpty()
                                ? new long[0]
                                : Arrays.stream(inner.split(","))
                                        .map(String::trim)
                                        .mapToLong(Long::parseLong)
                                        .toArray();
                params.add(new NamedAttribute(m.group(1), DenseArrayAttr.i64(values)));
            } else {
                params.add(new NamedAttribute(m.group(1), IntegerAttr.i64(Long.parseLong(value))));
            }
        }
        return params;
    }

    /** Parses {@code [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]}. */
    public static ConvDimensionNumbers parseCompact(String text) {
        String s = text.replaceAll("\\s+", "");
        int x = s.indexOf("]x[");
        int arrow = s.indexOf("]->[");
        if (!s.startsWith("[") || x < 0 || arrow < 0 || !s.endsWith("]")) {
            throw new IllegalArgumentException("bad convolution dimension numbers: " + text);
        }
        String[] in = s.substring(1, x).split(",");
        String[] kernel = s.substring(x + 3, arrow).split(",");
        String[] out = s.substring(arrow + 4, s.length() - 1).split(",");
        Layout li = Layout.of(in, 'b', 'f');
        Layout lk = Layout.of(kernel, 'i', 'o');
        Layout lo = Layout.of(out, 'b', 'f');
        return new ConvDimensionNumbers(
                li.first,
                li.second,
                li.spatial,
                lk.first,
                lk.second,
                lk.spatial,
                lo.first,
                lo.second,
                lo.spatial);
    }

    private record Layout(long first, long second, long[] spatial) {
        static Layout of(String[] labels, char firstLabel, char secondLabel) {
            long first = -1;
            long second = -1;
            if (labels.length < 2) {
                throw new IllegalArgumentException(
                        "too few dimensions in " + Arrays.toString(labels));
            }
            long[] spatial = new long[labels.length - 2];
            Arrays.fill(spatial, -1);
            for (int pos = 0; pos < labels.length; pos++) {
                String label = labels[pos];
                if (label.equals(String.valueOf(firstLabel)) && first < 0) {
                    first = pos;
                } else if (label.equals(String.valueOf(secondLabel)) && second < 0) {
                    second = pos;
                } else if (label.matches("\\d+")
                        && Integer.parseInt(label) < spatial.length
                        && spatial[Integer.parseInt(label)] < 0) {
                    spatial[Integer.parseInt(label)] = pos;
                } else {
                    throw new IllegalArgumentException(
                            "unexpected or duplicate dimension '"
                                    + label
                                    + "' in "
                                    + Arrays.toString(labels));
                }
            }
            if (first < 0 || second < 0) {
                throw new IllegalArgumentException(
                        "missing '"
                                + firstLabel
                                + "' or '"
                                + secondLabel
                                + "' in "
                                + Arrays.toString(labels));
            }
            return new Layout(first, second, spatial);
        }

        String render(char firstLabel, char secondLabel) {
            String[] labels = new String[spatial.length + 2];
            labels[(int) first] = String.valueOf(firstLabel);
            labels[(int) second] = String.valueOf(secondLabel);
            for (int i = 0; i < spatial.length; i++) {
                labels[(int) spatial[i]] = Integer.toString(i);
            }
            return "[" + String.join(", ", labels) + "]";
        }
    }

    /** The compact body {@code [b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]}. */
    public String compact() {
        return new Layout(inputBatchDimension, inputFeatureDimension, inputSpatialDimensions)
                        .render('b', 'f')
                + "x"
                + new Layout(
                                kernelInputFeatureDimension,
                                kernelOutputFeatureDimension,
                                kernelSpatialDimensions)
                        .render('i', 'o')
                + "->"
                + new Layout(outputBatchDimension, outputFeatureDimension, outputSpatialDimensions)
                        .render('b', 'f');
    }

    /**
     * The attribute as MLIR prints it ({@code printConvolutionDimensions}): the compact layout when
     * every dimension is within its rank, otherwise the {@code raw ...} form.
     */
    public DialectAttr toAttribute() {
        return DialectAttr.ofRaw("stablehlo", "conv", isCompactPrintable() ? compact() : raw());
    }

    /** {@code isConvolutionDimensionsPrintable}: every index lies in [0, rank) of its group. */
    public boolean isCompactPrintable() {
        return inRank(inputSpatialDimensions.length + 2, inputBatchDimension, inputFeatureDimension)
                && inRank(inputSpatialDimensions.length + 2, inputSpatialDimensions)
                && inRank(
                        kernelSpatialDimensions.length + 2,
                        kernelInputFeatureDimension,
                        kernelOutputFeatureDimension)
                && inRank(kernelSpatialDimensions.length + 2, kernelSpatialDimensions)
                && inRank(
                        outputSpatialDimensions.length + 2,
                        outputBatchDimension,
                        outputFeatureDimension)
                && inRank(outputSpatialDimensions.length + 2, outputSpatialDimensions)
                && distinct(inputBatchDimension, inputFeatureDimension, inputSpatialDimensions)
                && distinct(
                        kernelInputFeatureDimension,
                        kernelOutputFeatureDimension,
                        kernelSpatialDimensions)
                && distinct(outputBatchDimension, outputFeatureDimension, outputSpatialDimensions);
    }

    private static boolean inRank(long rank, long... dims) {
        for (long d : dims) {
            if (d < 0 || d >= rank) {
                return false;
            }
        }
        return true;
    }

    /** The compact form puts one label per position, so positions must not repeat. */
    private static boolean distinct(long first, long second, long[] spatial) {
        java.util.Set<Long> seen = new java.util.HashSet<>();
        seen.add(first);
        if (!seen.add(second)) {
            return false;
        }
        for (long d : spatial) {
            if (!seen.add(d)) {
                return false;
            }
        }
        return true;
    }

    /** {@code raw input_batch_dimension = 0, ...} ({@code printConvolutionDimensionsRaw}). */
    public String raw() {
        return "raw input_batch_dimension = "
                + inputBatchDimension
                + ", input_feature_dimension = "
                + inputFeatureDimension
                + ", input_spatial_dimensions = "
                + list(inputSpatialDimensions)
                + ", kernel_input_feature_dimension = "
                + kernelInputFeatureDimension
                + ", kernel_output_feature_dimension = "
                + kernelOutputFeatureDimension
                + ", kernel_spatial_dimensions = "
                + list(kernelSpatialDimensions)
                + ", output_batch_dimension = "
                + outputBatchDimension
                + ", output_feature_dimension = "
                + outputFeatureDimension
                + ", output_spatial_dimensions = "
                + list(outputSpatialDimensions);
    }

    private static String list(long[] values) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < values.length; i++) {
            sb.append(i > 0 ? ", " : "").append(values[i]);
        }
        return sb.append(']').toString();
    }
}
