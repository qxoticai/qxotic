package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;

/**
 * @deprecated use {@link MemoryViews}; removed after the migration.
 */
@Deprecated
public final class MemoryHelpers {

    private MemoryHelpers() {}

    public static <B> MemoryView<B> full(
            MemoryDomain<B> domain, DataType dataType, long count, Number value) {
        return MemoryViews.full(domain, dataType, count, value);
    }

    public static <B> MemoryView<B> full(MemoryDomain<B> domain, long count, boolean boolValue) {
        return MemoryViews.full(domain, count, boolValue);
    }

    public static <B> MemoryView<B> full(MemoryDomain<B> domain, long count, byte byteValue) {
        return MemoryViews.full(domain, count, byteValue);
    }

    public static <B> MemoryView<B> full(MemoryDomain<B> domain, long count, short shortValue) {
        return MemoryViews.full(domain, count, shortValue);
    }

    public static <B> MemoryView<B> full(MemoryDomain<B> domain, long count, int intValue) {
        return MemoryViews.full(domain, count, intValue);
    }

    public static <B> MemoryView<B> full(MemoryDomain<B> domain, long count, long longValue) {
        return MemoryViews.full(domain, count, longValue);
    }

    public static <B> MemoryView<B> full(MemoryDomain<B> domain, long count, float floatValue) {
        return MemoryViews.full(domain, count, floatValue);
    }

    public static <B> MemoryView<B> full(MemoryDomain<B> domain, long count, double doubleValue) {
        return MemoryViews.full(domain, count, doubleValue);
    }

    public static <B> MemoryView<B> full(
            MemoryDomain<B> domain, DataType dataType, Shape shape, Number value) {
        return MemoryViews.full(domain, dataType, shape, value);
    }

    public static <B> MemoryView<B> ones(MemoryDomain<B> domain, DataType dataType, long count) {
        return MemoryViews.ones(domain, dataType, count);
    }

    public static <B> MemoryView<B> ones(MemoryDomain<B> domain, DataType dataType, Shape shape) {
        return MemoryViews.ones(domain, dataType, shape);
    }

    public static <B> MemoryView<B> zeros(MemoryDomain<B> domain, DataType dataType, long count) {
        return MemoryViews.zeros(domain, dataType, count);
    }

    public static <B> MemoryView<B> zeros(MemoryDomain<B> domain, DataType dataType, Shape shape) {
        return MemoryViews.zeros(domain, dataType, shape);
    }

    public static <B> MemoryView<B> arange(
            MemoryDomain<B> domain, DataType dataType, long endExclusive) {
        return MemoryViews.arange(domain, dataType, endExclusive);
    }
}
