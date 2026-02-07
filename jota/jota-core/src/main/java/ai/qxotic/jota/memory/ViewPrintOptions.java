package ai.qxotic.jota.memory;

import ai.qxotic.jota.DataType;
import java.util.Locale;
import java.util.Objects;

public final class ViewPrintOptions {

    public static final int DEFAULT_MAX_ELEMENTS = 64;
    public static final int DEFAULT_EDGE_ITEMS = 3;

    private static final DefaultValueFormatter DEFAULT_FORMATTER = new DefaultValueFormatter();

    private final int maxElements;
    private final int edgeItems;
    private final boolean includeValues;
    private final boolean includeMetadata;
    private final ValueFormatter formatter;

    public ViewPrintOptions(
            int maxElements,
            int edgeItems,
            boolean includeValues,
            boolean includeMetadata,
            ValueFormatter formatter) {
        if (maxElements < 0) {
            throw new IllegalArgumentException("maxElements must be >= 0");
        }
        if (edgeItems < 0) {
            throw new IllegalArgumentException("edgeItems must be >= 0");
        }
        this.maxElements = maxElements;
        this.edgeItems = edgeItems;
        this.includeValues = includeValues;
        this.includeMetadata = includeMetadata;
        this.formatter = Objects.requireNonNullElse(formatter, DEFAULT_FORMATTER);
    }

    public static ViewPrintOptions compact() {
        return new ViewPrintOptions(
                DEFAULT_MAX_ELEMENTS, DEFAULT_EDGE_ITEMS, true, true, DEFAULT_FORMATTER);
    }

    public static ViewPrintOptions metaOnly() {
        return new ViewPrintOptions(
                DEFAULT_MAX_ELEMENTS, DEFAULT_EDGE_ITEMS, false, true, DEFAULT_FORMATTER);
    }

    public static ViewPrintOptions valuesOnly() {
        return new ViewPrintOptions(
                DEFAULT_MAX_ELEMENTS, DEFAULT_EDGE_ITEMS, true, false, DEFAULT_FORMATTER);
    }

    public int maxElements() {
        return maxElements;
    }

    public int edgeItems() {
        return edgeItems;
    }

    public boolean includeValues() {
        return includeValues;
    }

    public boolean includeMetadata() {
        return includeMetadata;
    }

    public ValueFormatter formatter() {
        return formatter;
    }

    private static final class DefaultValueFormatter implements ValueFormatter {

        @Override
        public String format(DataType dataType, Object value) {
            if (value == null) {
                return "null";
            }
            if (value instanceof Float floatValue) {
                return formatFloatCompact(floatValue);
            }
            if (value instanceof Double doubleValue) {
                return formatFloatCompact(doubleValue);
            }
            return value.toString();
        }

        private static String formatFloatCompact(float value) {
            if (Float.isNaN(value)) {
                return "NaN";
            }
            if (value == Float.POSITIVE_INFINITY) {
                return "+INF";
            }
            if (value == Float.NEGATIVE_INFINITY) {
                return "-INF";
            }
            return formatDecimal(value);
        }

        private static String formatFloatCompact(double value) {
            if (Double.isNaN(value)) {
                return "NaN";
            }
            if (value == Double.POSITIVE_INFINITY) {
                return "+INF";
            }
            if (value == Double.NEGATIVE_INFINITY) {
                return "-INF";
            }
            return formatDecimal(value);
        }

        private static String formatDecimal(double value) {
            if (value > Long.MAX_VALUE || value < Long.MIN_VALUE) {
                return Double.toString(value);
            }
            if (value == Math.rint(value)) {
                return String.format(Locale.ROOT, "%d.", (long) value);
            }
            String formatted = String.format(Locale.ROOT, "%.6f", value);
            formatted = formatted.replaceAll("0+$", "");
            formatted = formatted.replaceAll("\\.$", ".");
            return formatted;
        }
    }
}
