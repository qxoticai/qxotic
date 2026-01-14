package com.qxotic.jota.memory;

public final class MemoryAccessChecks {

    public static void checkBounds(Memory<?> memory, long byteOffset, long byteSize) {
        checkBounds(byteOffset >= 0, "negative byte offset");
        checkBounds(byteSize >= 0, "negative byte size");
        checkBounds(byteOffset + byteSize <= memory.byteSize(), "out of bounds access");
    }

    public static void checkWriteable(Memory<?> memory) {
        checkReadOnly(!memory.isReadOnly(), "memory is read-only");
    }

    public static void checkAlignedValue(long value, int alignment) {
        if (alignment <= 1) {
            return;
        }
        checkAlignment((value & (alignment - 1)) == 0, "unaligned access");
    }

    public enum Mode {
        OFF,
        ASSERT,
        RUNTIME
    }

    private static final String MODE_PROPERTY = "jota.memory.checks";
    private static final Mode MODE = resolveMode(System.getProperty(MODE_PROPERTY));

    private MemoryAccessChecks() {
    }

    public static Mode mode() {
        return MODE;
    }

    public static void checkBounds(boolean condition, String message) {
        if (MODE == Mode.RUNTIME) {
            if (!condition) {
                throw new IndexOutOfBoundsException(message);
            }
        } else if (MODE == Mode.ASSERT) {
            assert condition : message;
        }
    }

    public static void checkReadOnly(boolean condition, String message) {
        if (MODE == Mode.RUNTIME) {
            if (!condition) {
                throw new UnsupportedOperationException(message);
            }
        } else if (MODE == Mode.ASSERT) {
            assert condition : message;
        }
    }

    public static void checkAlignment(boolean condition, String message) {
        if (MODE == Mode.RUNTIME) {
            if (!condition) {
                throw new IllegalArgumentException(message);
            }
        } else if (MODE == Mode.ASSERT) {
            assert condition : message;
        }
    }

    private static Mode resolveMode(String raw) {
        if (raw == null || raw.isBlank()) {
            return Mode.RUNTIME;
        }
        return switch (raw.toLowerCase()) {
            case "off" -> Mode.OFF;
            case "assert" -> Mode.ASSERT;
            case "runtime" -> Mode.RUNTIME;
            default -> Mode.RUNTIME;
        };
    }
}
