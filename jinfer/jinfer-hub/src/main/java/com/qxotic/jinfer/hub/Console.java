package com.qxotic.jinfer.hub;

/**
 * The console probes behind {@link TerminalSupport}, one provider per runtime: on a JVM {@link
 * FfmConsole}, which links libc and kernel32 through the Foreign Function API and is the service
 * provider this module registers; a native image registers its own and leaves that one out.
 * Descriptors are 0, 1 or 2; a probe that cannot be made answers false or 0, never throws.
 */
public interface Console {
    /** Whether the descriptor is a terminal. */
    boolean isTerminal(int fd);

    /** The terminal's width in columns, 0 when unknown. */
    int columns(int fd);

    /**
     * ANSI sequences accepted: a Windows console's virtual-terminal mode set; on POSIX, whether the
     * descriptor is a terminal.
     */
    boolean enableAnsi(int fd);

    /**
     * A Windows console's output code page set to UTF-8, the original restored at exit; false
     * elsewhere, and when it cannot be set.
     */
    boolean enableUtf8();
}
