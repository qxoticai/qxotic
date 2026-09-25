package com.qxotic.jinfer.codecs;

import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.media.Media;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Base64;

/** Local audio playback through afplay, aplay, ffplay or Windows SoundPlayer. */
public final class AudioPlayer implements AutoCloseable {
    private Process process;
    private String[] command;
    private Path wav;
    private boolean firstClip = true;

    private AudioPlayer() {}

    /** Plays a complete waveform, waiting for playback and removing its temporary WAV. */
    public static void play(Media.Audio audio) throws IOException {
        try (var player = new AudioPlayer()) {
            player.playClip(audio);
        }
    }

    /** Plays clips in order, notifying the caller when the first clip reaches the player. */
    public static void stream(
            SpeechSynthesisModel<?, ?, ?> model,
            String text,
            SpeechOptions options,
            Runnable firstAudio)
            throws IOException {
        int sampleRate = model.sampleRate();
        if (sampleRate <= 0) throw new IOException("model does not report its audio sample rate");
        boolean wavClips = streamsWav(System.getProperty("os.name", ""));
        try (var player = new AudioPlayer()) {
            if (!wavClips) {
                String rate = Integer.toString(sampleRate);
                player.start(
                        new String[] {"aplay", "-q", "-f", "S16_LE", "-r", rate, "-c", "1", "-"},
                        new String[] {
                            "ffplay",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-f",
                            "s16le",
                            "-sample_rate",
                            rate,
                            "-ch_layout",
                            "mono",
                            "-nodisp",
                            "-autoexit",
                            "-"
                        });
            }
            model.speak(
                    text,
                    options,
                    clip -> {
                        if (clip.sampleRate() != sampleRate)
                            throw new IllegalArgumentException(
                                    "model reported "
                                            + sampleRate
                                            + " Hz but produced "
                                            + clip.sampleRate()
                                            + " Hz");
                        try {
                            if (wavClips) player.playClip(clip);
                            else if (!player.offer(AudioCodec.pcm16(clip))) return false;
                        } catch (IOException e) {
                            throw new UncheckedIOException(e);
                        }
                        if (player.firstClip) {
                            player.firstClip = false;
                            firstAudio.run();
                        }
                        return true;
                    });
        } catch (UncheckedIOException e) {
            throw e.getCause();
        }
    }

    private void playClip(Media.Audio audio) throws IOException {
        // The process plays while the next clip is synthesized. Wait here before replacing it:
        // at most one clip is ahead, and a player failure stops the stream at the next handoff.
        close();
        byte[] bytes = AudioCodec.wav(audio);
        wav = Files.createTempFile("jinfer-tts-", ".wav");
        Files.write(wav, bytes);
        start(fileCommands(System.getProperty("os.name", ""), wav));
    }

    static boolean streamsWav(String osName) {
        return osName.startsWith("Mac") || osName.startsWith("Windows");
    }

    static String[][] fileCommands(String osName, Path wav) {
        String file = wav.toAbsolutePath().toString();
        String[] nativePlayer;
        if (osName.startsWith("Windows")) {
            // EncodedCommand preserves Unicode and quoting through Windows' command-line parser.
            // Load explicitly: a failed load must report an error rather than play a system beep.
            String script =
                    "$ErrorActionPreference = 'Stop'; "
                            + "$player = New-Object System.Media.SoundPlayer; "
                            + "try { $player.SoundLocation = '"
                            + file.replace("'", "''")
                            + "'; $player.Load(); $player.PlaySync(); } finally {"
                            + " $player.Dispose(); }";
            nativePlayer =
                    new String[] {
                        "powershell.exe",
                        "-NoProfile",
                        "-NonInteractive",
                        "-EncodedCommand",
                        Base64.getEncoder()
                                .encodeToString(script.getBytes(StandardCharsets.UTF_16LE))
                    };
        } else if (osName.startsWith("Mac")) {
            nativePlayer = new String[] {"afplay", file};
        } else {
            nativePlayer = new String[] {"aplay", "-q", file};
        }
        return new String[][] {
            nativePlayer,
            {"ffplay", "-hide_banner", "-loglevel", "error", "-nodisp", "-autoexit", file}
        };
    }

    private void start(String[]... candidates) throws IOException {
        for (String[] candidate : candidates) {
            try {
                process =
                        new ProcessBuilder(candidate)
                                .redirectOutput(ProcessBuilder.Redirect.DISCARD)
                                .redirectError(ProcessBuilder.Redirect.INHERIT)
                                .start();
                command = candidate;
                return;
            } catch (IOException ignored) {
                // Only a launch failure tries another player; a player's nonzero exit is an error.
            }
        }
        throw new IOException(
                "no supported audio player found; tried "
                        + Arrays.stream(candidates).map(candidate -> candidate[0]).toList());
    }

    private boolean offer(byte[] pcm) {
        try {
            OutputStream pipe = process.getOutputStream();
            pipe.write(pcm);
            pipe.flush();
            return true;
        } catch (IOException quit) {
            return false; // close() reports the player's exit status.
        }
    }

    public static final class Failed extends IOException {
        private final int status;

        private Failed(int status, String message) {
            super(message);
            this.status = status;
        }

        public int status() {
            return status;
        }
    }

    @Override
    public void close() throws IOException {
        try {
            if (process != null) {
                try {
                    process.getOutputStream().close();
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
        } finally {
            process = null;
            if (wav != null) {
                Path finished = wav;
                wav = null;
                Files.deleteIfExists(finished);
            }
        }
    }

    private int waitFor() {
        try {
            return process.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            process.destroyForcibly();
            return 130;
        }
    }
}
