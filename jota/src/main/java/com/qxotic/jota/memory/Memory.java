package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;

public interface Memory<B> {

    long byteSize();

    boolean isReadOnly();

    Device device();

    B base();

    boolean supportsDataType(DataType dataType);
}
