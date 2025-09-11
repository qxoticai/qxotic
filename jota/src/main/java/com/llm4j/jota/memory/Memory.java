package com.llm4j.jota.memory;

import com.llm4j.jota.Device;

public interface Memory<B> {

    long byteSize();

    boolean isReadOnly();

    Device device();

    B base();
}
