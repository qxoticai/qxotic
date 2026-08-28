// Playback through whatever audio player the system already has.
package com.qxotic.jinfer.examples.inflect2;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * A system audio player, found on PATH.
 *
 * <p>Two shapes exist and they are NOT interchangeable. {@code aplay} and {@code ffplay} read raw
 * S16LE from stdin, so synthesis overlaps playback and the first audio lands long before the
 * sentence ends. {@code afplay} seeks, so a pipe fails outright ({@code AudioFileOpen failed}) and
 * it has to be handed a finished file - which is why a stock macOS, where it is usually the only
 * player installed, hears nothing until the last word is synthesized. Streaming is preferred
 * wherever it exists, for the latency.
 */
final class Player implements AutoCloseable {

    private final String[] command;
    private final Process process;
    private final Said said;
    private final OutputStream pipe;

    Player(String... command) throws IOException {
        this.command = command;
        // Keep whatever the player says. Discarding it costs nothing until the player refuses to
        // start, and then it costs everything: the only symptom left is a "Stream closed" from our
        // end of a pipe nobody is holding.
        this.process = new ProcessBuilder(command).redirectErrorStream(true).start();
        this.said = Said.draining(process.getInputStream());
        this.pipe = process.getOutputStream();
    }

    /**
     * A running player that takes raw S16LE on stdin, or null when none is installed.
     *
     * <p>The ffplay form uses the pcm demuxer's own {@code sample_rate} and {@code ch_layout}
     * rather than the {@code -ar}/{@code -ac} shorthands: ffplay 9 removed {@code -ac}, and with a
     * discarded stderr that turned every {@code --play} on a current ffmpeg into silence plus an
     * unrelated-looking pipe error. Demuxer options are the stable spelling.
     */
    static Player streaming(int rate) throws IOException {
        if (onPath("aplay"))
            return new Player(("aplay -f S16_LE -r " + rate + " -c 1 -").split(" "));
        if (onPath("ffplay"))
            return new Player(
                    ("ffplay -hide_banner -loglevel error -f s16le -sample_rate "
                                    + rate
                                    + " -ch_layout mono -nodisp -autoexit -")
                            .split(" "));
        return null;
    }

    /**
     * Throws unless a finished file can be played. Worth asking BEFORE synthesizing one: a missing
     * player should cost a message, not a whole utterance rendered into a temporary file that
     * nothing will ever open.
     */
    static void requireFilePlayer() throws IOException {
        if (!onPath("afplay"))
            throw new IOException("no audio player found - install ffplay, or aplay on Linux");
    }

    /** Play a finished WAV through to the end: the fallback for a player that cannot stream. */
    static void play(Path wav) throws IOException {
        requireFilePlayer();
        try (Player player = new Player("afplay", wav.toString())) {
            player.pipe.close(); // afplay reads the file, never stdin
        }
    }

    /** Hand over PCM; false once the pipe is gone, which cancels the synthesis. */
    boolean offer(byte[] pcm) {
        try {
            pipe.write(pcm);
            pipe.flush();
            return true;
        } catch (IOException quit) {
            return false;
        }
    }

    /**
     * A player that exited badly. Its status becomes the CLI's, so a script can tell a refused
     * format from a missing file, and its own words ride along in the message.
     */
    static final class Failed extends IOException {
        final int status;

        private Failed(int status, String message) {
            super(message);
            this.status = status;
        }
    }

    /** Wait for the player, and turn a bad exit into a message that quotes what it said. */
    @Override
    public void close() throws IOException {
        // Closing flushes, and flushing into a player that already exited throws. That the far end
        // is gone is how a player says "I quit"; the exit status says whether it ever ran at all.
        try {
            pipe.close();
        } catch (IOException gone) {
            // the exit status below has the real story
        }
        int status = waitFor();
        if (status != 0)
            throw new Failed(
                    status,
                    "audio player exited with status "
                            + status
                            + "\n  command: "
                            + String.join(" ", command)
                            + (said.isEmpty() ? "" : "\n  said: " + said));
    }

    private int waitFor() {
        try {
            return process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            process.destroy();
            return 130; // interrupted, by the shell's convention
        }
    }

    private static boolean onPath(String command) {
        String path = System.getenv("PATH");
        if (path == null) return false;
        for (String directory : path.split(File.pathSeparator))
            if (Files.isExecutable(Path.of(directory, command))) return true;
        return false;
    }

    /**
     * Whatever the player said, drained on a daemon thread so it can never block on a full pipe,
     * and kept only to quote back if it exits badly. At this log level a working player is silent.
     */
    private static final class Said {
        private static final int LIMIT = 500; // a broken player could talk forever
        private final StringBuilder text = new StringBuilder();

        static Said draining(InputStream from) {
            Said said = new Said();
            Thread reader = new Thread(() -> said.readAll(from), "player-output");
            reader.setDaemon(true);
            reader.start();
            return said;
        }

        private void readAll(InputStream from) {
            try (var in = new BufferedReader(new InputStreamReader(from, StandardCharsets.UTF_8))) {
                // ffplay paints its status line with CSI escapes and ends it with \r, which
                // readLine splits on too: strip them, or a silent player appears to have spoken.
                for (String line; (line = in.readLine()) != null; )
                    append(line.replaceAll("\\x1B\\[[\\d;?]*[ -/]*[@-~]", "").strip());
            } catch (IOException closed) {
                // the player is gone; whatever we have is what we quote
            }
        }

        private synchronized void append(String line) {
            if (line.isEmpty() || text.length() > LIMIT) return;
            text.append(text.isEmpty() ? "" : "; ").append(line);
        }

        synchronized boolean isEmpty() {
            return text.isEmpty();
        }

        @Override
        public synchronized String toString() {
            return text.toString();
        }
    }
}
