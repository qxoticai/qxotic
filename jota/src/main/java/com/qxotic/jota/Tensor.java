package com.qxotic.jota;

public interface Tensor {

    Layout layout();

    DataType dataType();

    default Shape shape() {
        return layout().shape();
    }

    default Stride stride() {
        return layout().stride();
    }

    default long size() {
        return shape().size();
    }

    default boolean isScalar() {
        return size() == 1;
    }

    // Realized and cache the Tensor, returns this
    Tensor realize();
}
