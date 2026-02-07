package ai.qxotic.jota;

import ai.qxotic.jota.impl.NestedTuple;
import ai.qxotic.jota.impl.ShapeFactory;

public interface Shape extends NestedTuple<Shape> {

    long size(int _modeIndex);

    long size();

    Shape flattenModes();

    default boolean hasZeroElements() {
        return size() == 0;
    }

    default boolean hasOneElement() {
        return size() == 1;
    }

    static Shape flat(long... dims) {
        return ShapeFactory.flat(dims);
    }

    static Shape of(Object... elements) {
        return ShapeFactory.of(elements);
    }

    static Shape pattern(String pattern, long... dims) {
        return ShapeFactory.pattern(pattern, dims);
    }

    static Shape template(NestedTuple<?> template, long... dims) {
        return ShapeFactory.template(template, dims);
    }

    static Shape scalar() {
        return ShapeFactory.scalar();
    }

    static Shape shape(Object... elements) {
        return of(elements);
    }
}
