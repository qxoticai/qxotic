package com.qxotic.jota.stablehlo.attr;

import java.util.ArrayList;
import java.util.List;

/** Helpers to read the integer parameters of dimension-number attributes. */
final class DimensionNumbers {

    private DimensionNumbers() {}

    static DialectAttr expect(Attribute attribute, String mnemonic) {
        if (attribute instanceof DialectAttr d
                && d.dialect().equals("stablehlo")
                && d.mnemonic().equals(mnemonic)
                && d.hasParams()) {
            return d;
        }
        throw new IllegalArgumentException(
                "expected #stablehlo." + mnemonic + "<...>, got " + attribute);
    }

    static long[] longs(DialectAttr d, String name) {
        return d.param(name).map(DimensionNumbers::toLongs).orElse(new long[0]);
    }

    static long scalar(DialectAttr d, String name, long defaultValue) {
        return d.param(name).map(a -> toLongs(a)[0]).orElse(defaultValue);
    }

    static long[] toLongs(Attribute a) {
        return switch (a) {
            case IntegerAttr i -> new long[] {i.value()};
            case DenseArrayAttr arr -> arr.values();
            case ArrayAttr arr -> {
                long[] out = new long[arr.elements().size()];
                for (int i = 0; i < out.length; i++) {
                    out[i] = toLongs(arr.elements().get(i))[0];
                }
                yield out;
            }
            default -> throw new IllegalArgumentException("expected integers, got " + a);
        };
    }

    static void add(List<NamedAttribute> params, String name, long[] values) {
        if (values.length > 0) {
            params.add(new NamedAttribute(name, DenseArrayAttr.i64(values)));
        }
    }

    static List<NamedAttribute> list() {
        return new ArrayList<>();
    }
}
