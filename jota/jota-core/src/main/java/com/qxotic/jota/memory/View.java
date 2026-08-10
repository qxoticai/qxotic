package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;

public interface View {
    Storage storage();

    Layout layout();

    DataType dataType();

    default Shape shape() {
        return layout().shape();
    }

    /**
     * The element-dimensioned shape: identical to {@link #shape()} except for block-quantized
     * dtypes, whose physical shape counts storage blocks — see {@link
     * DataType#logicalShape(Shape)}.
     */
    default Shape logicalShape() {
        return dataType().logicalShape(shape());
    }

    /** {@code logicalShape().size()} — the element count. */
    default long logicalSize() {
        return logicalShape().size();
    }

    default Stride stride() {
        return layout().stride();
    }
}
