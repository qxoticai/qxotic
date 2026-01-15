package ai.qxotic.jota.impl;

import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;

public final class LayoutFactory {

    public static Layout of(Shape shape, Stride stride) {
        return LayoutImpl.of(shape, stride);
    }

    public static Layout scalar() {
        return LayoutImpl.scalar();
    }
}

