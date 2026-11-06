package com.qxotic.jinfer.hub;

/**
 * The console probes behind {@link TerminalSupport}, one provider per runtime: on a JVM {@link
 * FfmConsole}, which links libc and kernel32 through the Foreign Function API and is the service
 * provider this module registers; a native image leaves that service out and {@link
 * TerminalSupport#use gives} its own at start. A provider overrides what it can probe; the defaults
 * are no console. Descriptors are 0, 1 or 2; a probe that cannot be made answers false or 0, never
 * throws.
 */
public interface Console {
    /** Whether the descriptor is a terminal. */
    default boolean isTerminal(int fd) {
        return false;
    }

    /** The terminal's width in columns, 0 when unknown. */
    default int columns(int fd) {
        return 0;
    }

    /**
     * ANSI sequences accepted: a Windows console's virtual-terminal mode set; on POSIX, whether the
     * descriptor is a terminal.
     */
    default boolean enableAnsi(int fd) {
        return false;
    }

    /**
     * A Windows console's output code page set to UTF-8, the original restored at exit; false
     * elsewhere, and when it cannot be set.
     */
    default boolean enableUtf8() {
        return false;
    }
}
