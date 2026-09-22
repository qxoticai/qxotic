package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.Transcription;
import java.io.PrintStream;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.IntSupplier;
import java.util.function.IntUnaryOperator;

/**
 * The live view of a streaming transcription on a terminal. Final words settle into the scrollback
 * in the terminal's own color, doubtful ones flagged; each final piece lands as a flowing gradient
 * of the {@link Theme} that fades into the text within a second. The provisional tail follows in
 * grey italics with a band of light sweeping through it, and a small waveform after it moves with
 * the voice. Below a blank line, a status line: a dot glowing while speech comes in, the elapsed
 * time, the input level, and "listening" while all is quiet.
 *
 * <p>The view paints itself 30 times a second on its own thread, so it stays live while the model
 * decodes; the caller only reports the input level and the transcript as they change. Text wraps at
 * the terminal's current width: lines in the scrollback stay as printed, and the live region below
 * them is measured again at the new width, as the terminal reflows it, so a resize repaints
 * cleanly. Colors go as deep as the terminal does; without any, attributes carry the cues. Each
 * frame is one synchronized write, and the cursor stays hidden while live.
 */
final class TranscriptHud {

    private static final int MAX_LIVE_ROWS = 6, METER_CELLS = 16, WAVE_BARS = 5;
    private static final long FRAME_MILLIS = 33, VOICE_NANOS = 400_000_000L;
    private static final long QUIET_NANOS = 1_500_000_000L; // before the view says it is listening
    private static final double DOUBT = 0.5; // final words below this confidence are flagged
    private static final double VOICE = 0.5; // levels above this point of their range are speech
    private static final double RANGE_DB = 30; // the level's range covers at least this much
    private static final double LAND_SECONDS = 1.1, SHIMMER_SECONDS = 1.8, BREATH_SECONDS = 1.6;
    private static final double SHIMMER_BAND = 5; // columns
    private static final int TAIL = 0x696E80, GLOW = 0xDCE6FF, SETTLE = 0xE1E1E6;
    private static final String RESET = "\u001b[0m", BOLD = "\u001b[1m", DIM = "\u001b[2m";
    private static final String ITALIC = "\u001b[3m", UNDERLINE = "\u001b[4m", GREY = "\u001b[90m";
    private static final String SYNC_BEGIN = "\u001b[?2026h", SYNC_END = "\u001b[?2026l";
    private static final String HIDE_CURSOR = "\u001b[?25l", SHOW_CURSOR = "\u001b[?25h";
    private static final String STOP_HINT = "Ctrl-C to stop";
    // the xterm defaults of the 16 basic colors: black, red, green, yellow, blue, magenta, cyan,
    // white, then their bright variants
    private static final int[] BASIC = {
        0x000000, 0xCD0000, 0x00CD00, 0xCDCD00, 0x0000EE, 0xCD00CD, 0x00CDCD, 0xE5E5E5,
        0x7F7F7F, 0xFF0000, 0x00FF00, 0xFFFF00, 0x5C5CFF, 0xFF00FF, 0x00FFFF, 0xFFFFFF
    };

    /**
     * The view's colors, as 0xRRGGBB: its gradients run from {@code low} to {@code high}, doubtful
     * final words read in {@code doubt}, and the status dot glows in {@code dot}.
     */
    record Theme(String name, int low, int high, int doubt, int dot) {
        static final List<Theme> BUNDLED =
                List.of(
                        new Theme("mint", 0x5EEAD4, 0xFACC15, 0xFB923C, 0xF87171),
                        new Theme("nord", 0x88C0D0, 0xEBCB8B, 0xD08770, 0xBF616A),
                        new Theme("catppuccin", 0x94E2D5, 0xFAB387, 0xF9E2AF, 0xF38BA8),
                        new Theme("ember", 0xFFC478, 0xFF8A65, 0x89B4FA, 0xE64553),
                        new Theme("frost", 0x5EEAD4, 0x7DD3FC, 0xFB923C, 0xF87171),
                        new Theme("mono", 0x80808C, 0xF5F5FA, 0xFB923C, 0xF87171));

        /** The bundled theme of that name, or null. */
        static Theme named(String name) {
            return BUNDLED.stream()
                    .filter(theme -> theme.name.equals(name))
                    .findFirst()
                    .orElse(null);
        }

        static List<String> names() {
            return BUNDLED.stream().map(Theme::name).toList();
        }
    }

    /** How many colors the terminal shows. */
    enum ColorDepth {
        NONE,
        BASIC, // the 16 ANSI colors
        EXTENDED, // the 256-color palette
        TRUE; // 24-bit

        /**
         * From the environment: none under {@code NO_COLOR}, 24-bit where {@code COLORTERM} says
         * so, 256 where {@code TERM} does, else the basic 16 every ANSI terminal has.
         */
        static ColorDepth of(Map<String, String> env) {
            if (!env.getOrDefault("NO_COLOR", "").isEmpty()) return NONE;
            String colorTerm = env.getOrDefault("COLORTERM", "").toLowerCase(Locale.ROOT);
            if (colorTerm.equals("truecolor") || colorTerm.equals("24bit")) return TRUE;
            if (env.getOrDefault("TERM", "").contains("256")) return EXTENDED;
            return BASIC;
        }
    }

    /** What the view draws with, and ASCII stand-ins for terminals without Unicode. */
    private record Glyphs(
            String eighths,
            String bars,
            String edges,
            char dot,
            char stop,
            char dash,
            String more) {
        static final Glyphs UNICODE = new Glyphs(" ▏▎▍▌▋▊▉█", "▁▂▃▄▅▆▇█", "▕▏", '●', '■', '·', "…");
        static final Glyphs ASCII = new Glyphs("    #####", "_.,-=+*#", "[]", '*', '#', '-', "...");
    }

    private final PrintStream out;
    private final IntSupplier columns;
    private final Theme theme;
    private final ColorDepth depth;
    private final Glyphs glyphs;
    private final long started = System.nanoTime();
    private final float[] levels = new float[80]; // input RMS of the last ~10 s, oldest first
    private List<Transcription.Token> committed = List.of(), tail = List.of();
    private int finalWords; // words in the committed tokens
    private int landedFrom; // where the latest final piece starts
    private long landedAt, voiceAt, shownAt; // when it arrived, speech was heard, text updated
    private int flushedWords; // in the scrollback, never repainted
    private int[] liveRows; // visible lengths of the last frame's live rows; null before any
    private String shown = ""; // the last frame, to skip repeats
    private int width, paintedWidth; // the terminal's columns this frame, and at the last paint
    private double loud, wave; // the level and the waveform's height, eased frame to frame
    private boolean finished;

    /** {@code columns} reports the terminal's current width. */
    TranscriptHud(Terminal terminal, IntSupplier columns, Theme theme) {
        this.out = terminal.out();
        this.columns = columns;
        this.theme = theme;
        this.depth = terminal.depth();
        this.glyphs = terminal.unicode() ? Glyphs.UNICODE : Glyphs.ASCII;
        this.landedAt = this.shownAt = started;
        this.voiceAt = started - VOICE_NANOS; // no speech heard yet
        // an interrupted session must not leave the terminal without a cursor
        Thread restore = new Thread(() -> out.append(RESET + SHOW_CURSOR).flush());
        Runtime.getRuntime().addShutdownHook(restore);
        Thread.ofPlatform().daemon().name("jinfer-transcript-view").start(this::run);
    }

    /** Records the RMS of the latest chunk of input. */
    synchronized void level(float rms) {
        System.arraycopy(levels, 1, levels, 0, levels.length - 1);
        levels[levels.length - 1] = rms;
        if (loudness(rms) > VOICE) voiceAt = System.nanoTime();
    }

    /**
     * Shows the {@code committed} tokens, then the provisional {@code tail}; tail tokens a newer
     * commit already covers are dropped.
     */
    synchronized void show(List<Transcription.Token> committed, List<Transcription.Token> tail) {
        this.committed = List.copyOf(committed);
        if (committed.isEmpty()) this.tail = List.copyOf(tail);
        else {
            Duration last = committed.getLast().start();
            this.tail = tail.stream().filter(token -> token.start().compareTo(last) > 0).toList();
        }
        int count = words(this.committed).size();
        if (count > finalWords) {
            landedFrom = finalWords;
            finalWords = count;
            landedAt = System.nanoTime();
        }
        shownAt = System.nanoTime();
    }

    /** Whether speech was heard after {@code nanos}, a {@link System#nanoTime} reading. */
    synchronized boolean heardSince(long nanos) {
        return voiceAt > nanos;
    }

    /** Clears the live region, settles the full transcript and closes with a summary line. */
    synchronized void finish(List<Transcription.Word> words) {
        finished = true;
        int columns = this.columns.getAsInt();
        StringBuilder frame = new StringBuilder(SYNC_BEGIN).append(restore(columns));
        int index = flushedWords;
        for (List<Transcription.Word> line : wrap(words.subList(index, words.size()), columns)) {
            if (!line.isEmpty()) frame.append(styled(line, index, words.size(), 0, 1)).append('\n');
            index += line.size();
        }
        long seconds = (System.nanoTime() - started) / 1_000_000_000L;
        String summary = "%c %s %c %d words";
        frame.append('\n').append(grey());
        frame.append(summary.formatted(glyphs.stop(), clock(seconds), glyphs.dash(), words.size()));
        out.append(frame).append(RESET + '\n' + SHOW_CURSOR + SYNC_END).flush();
    }

    private void run() {
        try {
            while (paint()) Thread.sleep(FRAME_MILLIS);
        } catch (InterruptedException stopped) {
            Thread.currentThread().interrupt();
        }
    }

    /** Paints one frame; false once finished. */
    private synchronized boolean paint() {
        if (finished) return false;
        long now = System.nanoTime();
        double t = (now - started) / 1e9;
        int columns = this.columns.getAsInt();
        width = columns;
        loud = ease(loud, loudness(levels[levels.length - 1]), 0.55, 0.12);
        // pieces end mid-word, so words are grouped over all the tokens; a final word that runs on
        // into the tail is still provisional
        List<Transcription.Token> tokens = new ArrayList<>(committed);
        tokens.addAll(tail);
        List<Transcription.Word> words = words(tokens);
        boolean runsOn = !tail.isEmpty() && !tail.getFirst().text().startsWith(" ");
        int provisionalFrom = runsOn ? finalWords - 1 : finalWords;
        double landing = (now - landedAt) / 1e9 / LAND_SECONDS; // from 0, landed at 1
        List<List<Transcription.Word>> lines =
                wrap(words.subList(flushedWords, words.size()), columns);

        // a line goes to the scrollback once all its words are final and done landing
        StringBuilder frame = new StringBuilder();
        int flushed = 0, index = flushedWords;
        int permanent = landing < 1 ? Math.min(landedFrom, provisionalFrom) : provisionalFrom;
        while (flushed < lines.size() - 1 && index + lines.get(flushed).size() <= permanent) {
            frame.append(styled(lines.get(flushed), index, provisionalFrom, t, 1)).append('\n');
            index += lines.get(flushed++).size();
        }
        int flushedIndex = index;
        List<String> live = new ArrayList<>();
        int from = Math.max(flushed, lines.size() - MAX_LIVE_ROWS);
        for (int i = flushed; i < lines.size(); index += lines.get(i++).size()) {
            if (i < from) continue;
            String more = i == from && from > flushed ? grey() + glyphs.more() + ' ' + RESET : "";
            String row = more + styled(lines.get(i), index, provisionalFrom, t, landing);
            // the waveform only where it fits: a row that overflows wraps, and the next frame
            // would climb back one row short of where it started
            String wave = i == lines.size() - 1 ? waveform(now, t) : "";
            boolean fits = visibleLength(row) + visibleLength(wave) <= columns - 1;
            live.add(fits ? row + wave : row);
        }
        live.add("");
        live.add(status(now, t, columns));
        String text = frame.append(String.join("\n", live)).toString();
        if (text.equals(shown) && columns == paintedWidth) return true;

        String hide = liveRows == null ? HIDE_CURSOR : "";
        out.append(SYNC_BEGIN + hide).append(restore(columns)).append(text).append(SYNC_END);
        out.flush();
        shown = text;
        paintedWidth = columns;
        flushedWords = flushedIndex;
        liveRows = live.stream().mapToInt(TranscriptHud::visibleLength).toArray();
        return true;
    }

    /**
     * The line's words from index {@code first}: the tail (from {@code provisionalFrom}) shimmers,
     * the latest final piece lands while {@code landing} is below 1, and doubtful words are
     * flagged. Colors follow the screen column, so a sweep runs across the rows.
     */
    private String styled(
            List<Transcription.Word> line,
            int first,
            int provisionalFrom,
            double t,
            double landing) {
        StringBuilder text = new StringBuilder();
        int column = 0;
        for (int i = 0; i < line.size(); i++) {
            Transcription.Word word = line.get(i);
            int index = first + i;
            boolean tail = index >= provisionalFrom;
            boolean lands = !tail && landing < 1 && index >= landedFrom;
            boolean doubtful = !tail && !lands && word.confidence() < DOUBT;
            String attributes;
            IntUnaryOperator color; // by column; null for the terminal's own
            if (depth == ColorDepth.NONE) {
                attributes = tail ? DIM + ITALIC : lands ? BOLD : doubtful ? UNDERLINE : "";
                color = null;
            } else if (tail) {
                attributes = ITALIC;
                color = x -> shimmer(x, t, width);
            } else if (lands) {
                attributes = landing < 0.35 ? BOLD : "";
                color = x -> mix(gradient(x / 40.0 - t * 0.6), SETTLE, smooth(landing));
            } else {
                attributes = "";
                color = doubtful ? x -> theme.doubt() : null;
            }
            if (i > 0) {
                text.append(' ');
                column++;
            }
            text.append(attributes);
            String last = "";
            for (int c : word.text().codePoints().toArray()) {
                String code = color == null ? "" : fg(color.applyAsInt(column));
                if (!code.equals(last)) text.append(last = code);
                text.appendCodePoint(c);
                column++;
            }
            if (!attributes.isEmpty() || color != null) text.append(RESET);
        }
        return text.toString();
    }

    /** Grey, lit up as the band of light sweeping across the terminal's columns passes. */
    private static int shimmer(int column, double t, int columns) {
        double center = fraction(t / SHIMMER_SECONDS) * (columns + 2 * SHIMMER_BAND) - SHIMMER_BAND;
        return mix(TAIL, GLOW, Math.exp(-Math.pow((column - center) / SHIMMER_BAND, 2)));
    }

    /**
     * The status line: a dot, solid while speech comes in and breathing otherwise, the elapsed
     * time, the level, "listening" while all is quiet, and the stop hint at the right edge where it
     * fits.
     */
    private String status(long now, double t, int columns) {
        boolean speaking = now - voiceAt < VOICE_NANOS;
        // a pause between sentences is not silence: the label waits for a real one
        boolean idle = now - voiceAt > QUIET_NANOS && voiceAt <= shownAt;
        int dot = speaking ? theme.dot() : mix(mix(0, theme.dot(), 0.4), theme.dot(), breath(t));
        String line =
                fg(dot)
                        + glyphs.dot()
                        + grey()
                        + ' '
                        + clock((now - started) / 1_000_000_000L)
                        + ' '
                        + meter()
                        + (idle ? "  listening" + glyphs.more() : "");
        int pad = columns - 1 - visibleLength(line) - STOP_HINT.length();
        return (pad >= 2 ? line + " ".repeat(pad) + STOP_HINT : line) + RESET;
    }

    /**
     * Bars after the text while speech comes in or is not shown yet, rippling with the level: low
     * colors at rest, high at the peaks. It eases in and out.
     */
    private String waveform(long now, double t) {
        boolean pending = voiceAt > shownAt || now - voiceAt < VOICE_NANOS;
        wave = ease(wave, pending ? Math.max(0.2, loud) : 0, 0.5, 0.15);
        if (wave < 0.04) return "";
        StringBuilder bars = new StringBuilder(" ");
        for (int i = 0; i < WAVE_BARS; i++) {
            double ripple = 0.5 + 0.5 * Math.sin(2 * Math.PI * t / 0.8 + i * 1.3);
            double height = Math.clamp(wave * (0.35 + 0.65 * ripple), 0, 1);
            bars.append(fg(mix(theme.low(), theme.high(), height)));
            bars.append(glyphs.bars().charAt((int) Math.round(height * 7)));
        }
        return bars.append(RESET).toString();
    }

    /** The eased level as a bar, its cells low to high colors along it. */
    private String meter() {
        int eighths = (int) Math.round(loud * METER_CELLS * 8);
        StringBuilder bar = new StringBuilder().append(glyphs.edges().charAt(0));
        for (int cell = 0; cell < METER_CELLS; cell++) {
            bar.append(fg(mix(theme.low(), theme.high(), cell / (METER_CELLS - 1.0))));
            bar.append(glyphs.eighths().charAt(Math.clamp(eighths - 8 * cell, 0, 8)));
        }
        return bar.append(grey()).append(glyphs.edges().charAt(1)).toString();
    }

    /**
     * Where {@code level} sits between the quietest level heard (0) and the loudest (1), across at
     * least {@link #RANGE_DB}, so steady noise reads low whatever the input gain. Unset (zero)
     * levels are ignored.
     */
    private double loudness(float level) {
        if (level <= 0) return 0;
        double floor = Double.MAX_VALUE, peak = -Double.MAX_VALUE;
        for (float each : levels) {
            if (each <= 0) continue;
            floor = Math.min(floor, decibels(each));
            peak = Math.max(peak, decibels(each));
        }
        return Math.clamp((decibels(level) - floor) / Math.max(peak - floor, RANGE_DB), 0, 1);
    }

    /**
     * Moves to the top of the last frame's live region and clears to the end of the screen. The
     * region's rows are counted at the current width, as a terminal that reflows on resize lays
     * them out; the cursor sits at the end of the last one.
     */
    private String restore(int columns) {
        if (liveRows == null) return "\r";
        int up = -1;
        for (int length : liveRows) up += Math.max(1, (length + columns - 1) / columns);
        return "\r" + (up > 0 ? "\u001b[" + up + 'A' : "") + "\u001b[0J";
    }

    /** The foreground escape for {@code rgb} at the terminal's depth; empty without colors. */
    String fg(int rgb) {
        return switch (depth) {
            case NONE -> "";
            case BASIC -> {
                int nearest = 0;
                for (int i = 1; i < BASIC.length; i++)
                    if (distance(rgb, BASIC[i]) < distance(rgb, BASIC[nearest])) nearest = i;
                yield "\u001b[" + (nearest < 8 ? 30 + nearest : 90 + nearest - 8) + 'm';
            }
            case EXTENDED -> {
                // the 6x6x6 cube at 16 + 36r + 6g + b, each channel on 6 levels
                int cube = 16;
                for (int shift = 16, weight = 36; shift >= 0; shift -= 8, weight /= 6)
                    cube += weight * (int) Math.round((rgb >> shift & 0xFF) * 5 / 255.0);
                yield "\u001b[38;5;" + cube + 'm';
            }
            case TRUE -> "\u001b[38;2;%d;%d;%dm".formatted(rgb >> 16, rgb >> 8 & 0xFF, rgb & 0xFF);
        };
    }

    /** For the chrome: bright black, or dim without colors. */
    private String grey() {
        return depth == ColorDepth.NONE ? DIM : GREY;
    }

    /** Low to high and back as {@code position} goes from 0 to 1; it wraps around. */
    private int gradient(double position) {
        return mix(theme.low(), theme.high(), (1 - Math.cos(2 * Math.PI * position)) / 2);
    }

    /** From 0 to 1 and back once a breath, smoothly. */
    private static double breath(double t) {
        return (1 - Math.cos(2 * Math.PI * t / BREATH_SECONDS)) / 2;
    }

    /** Eased from 0 to 1 over {@code x} in [0, 1]. */
    private static double smooth(double x) {
        x = Math.clamp(x, 0, 1);
        return x * x * (3 - 2 * x);
    }

    /** One frame's step from {@code value} toward {@code target}: {@code up} or {@code down}. */
    private static double ease(double value, double target, double up, double down) {
        return value + (target - value) * (target > value ? up : down);
    }

    private static int mix(int a, int b, double f) {
        int rgb = 0;
        for (int shift = 16; shift >= 0; shift -= 8) {
            int from = a >> shift & 0xFF, to = b >> shift & 0xFF;
            rgb |= (int) Math.round(from + (to - from) * f) << shift;
        }
        return rgb;
    }

    private static int distance(int a, int b) {
        int sum = 0;
        for (int shift = 16; shift >= 0; shift -= 8) {
            int d = (a >> shift & 0xFF) - (b >> shift & 0xFF);
            sum += d * d;
        }
        return sum;
    }

    private static double decibels(float level) {
        return 20 * Math.log10(level);
    }

    private static double fraction(double x) {
        return x - Math.floor(x);
    }

    private static String clock(long seconds) {
        return "%d:%02d".formatted(seconds / 60, seconds % 60);
    }

    private static int visibleLength(String row) {
        String plain = row.replaceAll("\u001b\\[[0-9;?]*[A-Za-z]", "");
        return plain.codePointCount(0, plain.length());
    }

    private static List<Transcription.Word> words(List<Transcription.Token> tokens) {
        return new Transcription("", tokens).words();
    }

    /**
     * Greedy word wrap, one column short of {@code columns} so a full line never triggers the
     * terminal's own wrap; a word longer than a line gets a line of its own.
     */
    private static List<List<Transcription.Word>> wrap(
            List<Transcription.Word> words, int columns) {
        List<List<Transcription.Word>> lines = new ArrayList<>();
        List<Transcription.Word> line = new ArrayList<>();
        int length = 0;
        for (Transcription.Word word : words) {
            int needed = line.isEmpty() ? word.text().length() : length + 1 + word.text().length();
            if (!line.isEmpty() && needed > columns - 1) {
                lines.add(line);
                line = new ArrayList<>();
                needed = word.text().length();
            }
            line.add(word);
            length = needed;
        }
        lines.add(line);
        return lines;
    }
}
