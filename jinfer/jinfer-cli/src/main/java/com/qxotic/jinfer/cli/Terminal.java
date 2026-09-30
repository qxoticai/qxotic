package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.hub.TerminalSupport;
import java.io.PrintStream;

/** The live view's output and palette; platform console access is shared with download progress. */
record Terminal(PrintStream out, boolean unicode, TranscriptHud.ColorDepth depth) {

    private static final boolean WINDOWS = System.getProperty("os.name").startsWith("Windows");

    static Terminal stderr() {
        return stderr(System.err, "auto");
    }

    static Terminal stderr(PrintStream out, String color) {
        if ("dumb".equals(System.getenv("TERM")) || !TerminalSupport.enableAnsi(2)) return null;
        TranscriptHud.ColorDepth depth =
                color.equals("off")
                        ? TranscriptHud.ColorDepth.NONE
                        : color.equals("on")
                                ? TranscriptHud.ColorDepth.TRUE
                                : TranscriptHud.ColorDepth.of(System.getenv());
        if (WINDOWS && depth != TranscriptHud.ColorDepth.NONE)
            depth = TranscriptHud.ColorDepth.TRUE;
        return new Terminal(out, TerminalSupport.enableUtf8(), depth);
    }

    static boolean isTerminal(int fd) {
        return TerminalSupport.isTerminal(fd);
    }

    static int columns() {
        return TerminalSupport.columns(2);
    }
}
