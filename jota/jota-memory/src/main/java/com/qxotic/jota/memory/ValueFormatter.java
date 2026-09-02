package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;

/** Formats one element value for {@link MemoryViewPrinter}. */
@FunctionalInterface
public interface ValueFormatter {

    String format(DataType dataType, Object value);
}
