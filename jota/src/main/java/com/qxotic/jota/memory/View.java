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

    default Stride stride() {
        return layout().stride();
    }
}
