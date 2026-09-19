package com.qxotic.jinfer.cli;

import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

/**
 * The live view of a streaming transcription on a terminal: committed text flows into the
 * scrollback as permanent wrapped lines, the still-revisable tail continues after it dimmed and
 * redraws in place, and a recording status line sits at the bottom. Plain ANSI on stderr; the
 * caller gates on stderr being a terminal.
 */
final class TranscriptHud {

    // ponytail: fixed 78-column wrap (no TIOCGWINSZ downcall); a narrower terminal wraps raggedly
    private static final int WIDTH = 78;
    private static final int MAX_VOLATILE_ROWS = 4;
    private static final String DIM = "\u001b[90m";
    private static final String RED = "\u001b[31m";
    private static final String RESET = "\u001b[0m";

    private final PrintStream out;
    private int flushedLines; // committed lines already in the scrollback, never repainted
    private int repaintUp = -1; // text rows above the status line on the last frame; -1 = none yet

    TranscriptHud(PrintStream out) {
        this.out = out;
    }

    /** Redraws the live view; {@code partial} extends {@code committed} (checked defensively). */
    void render(String committed, String partial, long seconds) {
        if (!partial.startsWith(committed)) committed = ""; // never trust, never garble
        StringBuilder frame = restore();
        List<String> lines = wrap(committed);
        for (int i = flushedLines; i < lines.size() - 1; i++)
            frame.append(lines.get(i)).append('\n');
        flushedLines = Math.max(flushedLines, lines.size() - 1);

        // The open committed line and the dim tail flow together, wrapped as one plain string;
        // the dim escape is inserted at the boundary after wrapping so the width math stays true.
        String open = lines.isEmpty() ? "" : lines.get(lines.size() - 1);
        String live = open + partial.substring(committed.length());
        List<String> volatileLines = wrap(live);
        int from = Math.max(0, volatileLines.size() - MAX_VOLATILE_ROWS);
        int offset = 0;
        for (int i = 0; i < from; i++) offset += volatileLines.get(i).length() + 1;
        int boundary = open.length();
        for (int i = from; i < volatileLines.size(); i++) {
            String line = volatileLines.get(i);
            if (i == from && from > 0) frame.append(DIM).append('…').append(RESET);
            if (boundary <= offset) frame.append(DIM).append(line);
            else if (boundary >= offset + line.length()) frame.append(line);
            else
                frame.append(line, 0, boundary - offset)
                        .append(DIM)
                        .append(line, boundary - offset, line.length());
            frame.append(RESET).append('\n');
            offset += line.length() + 1;
        }
        frame.append(RED)
                .append('●')
                .append(RESET)
                .append(DIM)
                .append(" %d:%02d".formatted(seconds / 60, seconds % 60))
                .append(RESET);
        repaintUp = volatileLines.size() - from;
        out.print(frame);
        out.flush();
    }

    /** Clears the live region and settles the full transcript as permanent flowed text. */
    void finish(String text) {
        StringBuilder frame = restore();
        List<String> lines = wrap(text);
        for (int i = flushedLines; i < lines.size(); i++) frame.append(lines.get(i)).append('\n');
        flushedLines = lines.size();
        out.print(frame);
        out.flush();
    }

    /** Moves to the top of the previous frame's repaintable region and clears to screen end. */
    private StringBuilder restore() {
        StringBuilder frame = new StringBuilder("\r");
        if (repaintUp > 0) frame.append("\u001b[").append(repaintUp).append('A');
        if (repaintUp >= 0) frame.append("\u001b[0J");
        return frame;
    }

    /** Greedy word wrap at {@link #WIDTH}; single-spaced input round-trips losslessly. */
    private static List<String> wrap(String text) {
        List<String> lines = new ArrayList<>();
        StringBuilder line = new StringBuilder();
        for (String word : text.split(" ", -1)) {
            while (word.length() > WIDTH) { // an unbroken run longer than the line
                lines.add(word.substring(0, WIDTH));
                word = word.substring(WIDTH);
            }
            if (line.isEmpty()) line.append(word);
            else if (line.length() + 1 + word.length() <= WIDTH) line.append(' ').append(word);
            else {
                lines.add(line.toString());
                line = new StringBuilder(word);
            }
        }
        lines.add(line.toString());
        return lines;
    }
}
