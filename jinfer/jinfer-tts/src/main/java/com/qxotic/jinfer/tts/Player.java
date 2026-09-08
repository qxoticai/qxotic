package com.qxotic.jinfer.tts;

import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Path;

final class Player implements AutoCloseable {
    private final String[] command;
    private final Process process;
    private final OutputStream pipe;

    private Player(String... command) throws IOException {
        this.command = command;
        process =
                new ProcessBuilder(command).redirectError(ProcessBuilder.Redirect.INHERIT).start();
        pipe = process.getOutputStream();
    }

    static boolean isMac() {
        return System.getProperty("os.name", "").startsWith("Mac");
    }

    static Player stream(int rate) {
        return firstThatRuns(aplay(rate), ffplay(rate));
    }

    static void play(Path wav) throws IOException {
        Player player = firstThatRuns(afplay(wav), soundPlayer(wav));
        if (player == null) throw new IOException("no supported audio player found");
        player.close();
    }

    private static Player firstThatRuns(String[]... candidates) {
        for (String[] command : candidates) {
            try {
                return new Player(command);
            } catch (IOException ignored) {
                // The operating system could not launch it; try the next player.
            }
        }
        return null;
    }

    static String[] aplay(int rate) {
        return ("aplay -f S16_LE -r " + rate + " -c 1 -").split(" ");
    }

    static String[] ffplay(int rate) {
        return ("ffplay -hide_banner -loglevel error -f s16le -sample_rate "
                        + rate
                        + " -ch_layout mono -nodisp -autoexit -")
                .split(" ");
    }

    static String[] afplay(Path wav) {
        return new String[] {"afplay", wav.toString()};
    }

    static String[] soundPlayer(Path wav) {
        return new String[] {
            "powershell",
            "-NoProfile",
            "-Command",
            "(New-Object Media.SoundPlayer '" + wav.toString().replace("'", "''") + "').PlaySync()"
        };
    }

    boolean offer(byte[] pcm) {
        try {
            pipe.write(pcm);
            pipe.flush();
            return true;
        } catch (IOException quit) {
            return false;
        }
    }

    static final class Failed extends IOException {
        final int status;

        private Failed(int status, String message) {
            super(message);
            this.status = status;
        }
    }

    @Override
    public void close() throws IOException {
        try {
            pipe.close();
        } catch (IOException ignored) {
            // The process status below is the useful error.
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
            return 130;
        }
    }
}
