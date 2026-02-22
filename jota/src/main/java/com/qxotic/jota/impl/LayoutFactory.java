package com.qxotic.jota.impl;

import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;

public final class LayoutFactory {

    public static Layout of(Shape shape, Stride stride) {
        return LayoutImpl.of(shape, stride);
    }

    public static Layout scalar() {
        return LayoutImpl.scalar();
    }
}
