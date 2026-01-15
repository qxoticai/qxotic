package ai.qxotic.jota.memory;

import ai.qxotic.jota.DataType;

@FunctionalInterface
public interface ValueFormatter {

    String format(DataType dataType, Object value);
}
