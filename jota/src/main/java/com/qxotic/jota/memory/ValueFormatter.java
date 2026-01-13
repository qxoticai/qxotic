package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;

@FunctionalInterface
public interface ValueFormatter {

    String format(DataType dataType, Object value);
}
