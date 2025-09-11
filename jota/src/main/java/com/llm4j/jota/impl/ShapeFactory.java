package com.llm4j.jota.impl;

import com.llm4j.jota.Shape;

public final class ShapeFactory {

    private ShapeFactory() {
    }

    public static Shape of(long... dimensions) {
        if (dimensions.length == 0) {
            return scalar();
        }
        return new ShapeImpl(dimensions);
    }

    public static Shape scalar() {
        return ShapeImpl.SCALAR;
    }
}
