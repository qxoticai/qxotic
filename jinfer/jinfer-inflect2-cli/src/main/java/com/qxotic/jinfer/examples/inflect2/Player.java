// Playback through whatever audio player the system already has.
package com.qxotic.jinfer.examples.inflect2;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;

/**
 * A system audio player.
 *
 * <p>Two shapes exist and they are NOT interchangeable. {@code aplay} and {@code ffplay} read raw
 * S16LE from stdin, so synthesis overlaps playback and the first audio lands long before the
 * sentence ends. {@code afplay} and PowerShell's {@code SoundPlayer} seek, so a pipe fails outright
 * ({@code AudioFileOpen failed}) and they have to be handed a finished file - which is why a stock
 * macOS or Windows hears nothing until the last word is synthesized. Streaming is preferred
 * wherever it exists, for the latency.
 *
 * <p>Every desktop has a way to be heard: Linux through aplay, macOS through afplay, Windows
 * through PowerShell, and any of them through ffplay when ffmpeg is installed.
 *
 * <p>Candidates are tried by RUNNING them. The OS already knows how to find a program on PATH -
 * including Windows appending {@code .exe} from PATHEXT - and {@code start()} says so by throwing.
 * Searching PATH ourselves would only be a second, worse copy of that, with its own bugs.
 */
final class Player implements AutoCloseable {

    private final String[] command;
    private final Process process;
    private final OutputStream pipe;

    Player(String... command) throws IOException {
        this.command = command;
        // The player's stderr is inherited: when it refuses the stream - no audio device, an
        // option this ffmpeg no longer has - its reason is the diagnosis, in its own words.
        this.process =
                new ProcessBuilder(command).redirectError(ProcessBuilder.Redirect.INHERIT).start();
        this.pipe = process.getOutputStream();
    }

    /** A running player that takes raw S16LE on stdin, or null when none is installed. */
    static Player streaming(int rate) {
        return firstThatRuns(aplay(rate), ffplay(rate));
    }

    /** Play a finished WAV through to the end: the fallback for a player that cannot stream. */
    static void play(Path wav) throws IOException {
        Player player = firstThatRuns(afplay(wav), soundPlayer(wav));
        if (player == null)
            throw new IOException("no audio player found - install ffplay, or aplay on Linux");
        // It reads the file, never stdin, so there is nothing to hand over: closing is the whole
        // interaction - it waits for the playback to finish and reports a bad exit.
        player.close();
    }

    /** The first candidate the OS can run; a program that is not installed is simply skipped. */
    private static Player firstThatRuns(String[]... candidates) {
        for (String[] command : candidates) {
            try {
                return new Player(command);
            } catch (IOException notInstalled) {
                // nothing to fall back FROM: the next candidate is the fallback
            }
        }
        return null;
    }

    static String[] aplay(int rate) {
        return ("aplay -f S16_LE -r " + rate + " -c 1 -").split(" ");
    }

    /**
     * ffplay reading the pipe, with the pcm demuxer's own {@code sample_rate} and {@code ch_layout}
     * rather than the {@code -ar}/{@code -ac} shorthands: ffplay 9 removed {@code -ac}, which
     * turned every {@code --play} on a current ffmpeg into silence. Demuxer options are the
     * spelling that survives a version.
     */
    static String[] ffplay(int rate) {
        return ("ffplay -hide_banner -loglevel error -f s16le -sample_rate "
                        + rate
                        + " -ch_layout mono -nodisp -autoexit -")
                .split(" ");
    }

    static String[] afplay(Path wav) {
        return new String[] {"afplay", wav.toString()};
    }

    /** Windows' own player. Only Windows has a {@code powershell}; pwsh elsewhere is not it. */
    static String[] soundPlayer(Path wav) {
        return new String[] {
            "powershell",
            "-NoProfile",
            "-Command",
            // PlaySync, or the shell exits and cuts the audio off mid-word. A quote in the path
            // is doubled, which is how PowerShell escapes one.
            "(New-Object Media.SoundPlayer '" + wav.toString().replace("'", "''") + "').PlaySync()"
        };
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
     * format from a missing device without parsing anything we print.
     */
    static final class Failed extends IOException {
        final int status;

        private Failed(int status, String message) {
            super(message);
            this.status = status;
        }
    }

    /** Wait for the player, and turn a bad exit into the status and the command that earned it. */
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
                            + String.join(" ", command));
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
}
