package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.codecs.AudioPlayer;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.media.Media;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;

/** Text to speech; playback is the default, and saving audio is explicit. */
final class Speak {
    private Speak() {}

    static final class Settings {
        Path output;
        Double speed;
        boolean play;
    }

    static boolean read(Options o, Options.Args a) {
        switch (a.name) {
            case "--play" -> o.speech.play = a.flag();
            case "--output", "-o" -> o.speech.output = Path.of(a.value());
            case "--speed" -> {
                String value = a.value();
                try {
                    o.speech.speed = Double.parseDouble(value);
                } catch (NumberFormatException e) {
                    throw new Options.Usage("--speed expects a number, got " + value);
                }
            }
            default -> {
                return false;
            }
        }
        o.use(a.name, Set.of("speak"));
        return true;
    }

    static void validate(Options o) {
        Options.require(
                o.input != null && !o.input.isBlank(),
                "speak requires non-blank text or '-' for stdin");
        Settings s = o.speech;
        Options.require(
                s.speed == null || (Double.isFinite(s.speed) && s.speed > 0),
                "--speed must be positive and finite; got %s",
                s.speed);
        Options.require(!s.play || !o.stream, "--play and --stream cannot be used together");
        Options.require(
                s.output == null || (!s.play && !o.stream),
                "--output cannot combine with --play or --stream");
        if (s.output == null && !s.play && !o.stream) {
            if (o.legacy) s.output = Path.of("output.wav");
            else s.play = true;
        }
    }

    /** The external playback boundary. Model behavior is tested through the existing model API. */
    interface Playback {
        void play(Media.Audio audio) throws IOException;

        void stream(
                SpeechSynthesisModel<?, ?, ?> model,
                String text,
                SpeechOptions options,
                Runnable firstAudio)
                throws IOException;
    }

    private static final Playback PLAYER =
            new Playback() {
                public void play(Media.Audio audio) throws IOException {
                    AudioPlayer.play(audio);
                }

                public void stream(
                        SpeechSynthesisModel<?, ?, ?> model,
                        String text,
                        SpeechOptions options,
                        Runnable firstAudio)
                        throws IOException {
                    AudioPlayer.stream(model, text, options, firstAudio);
                }
            };

    static int run(Options options, Main.IO io, ModelStore store) throws IOException {
        String text = io.text(options.input);
        Options.Files files = options.resolve(store);
        Arena arena = Arenas.newCrossThread();
        try {
            SpeechSynthesisModel<?, ?, ?> model;
            try (var spinner = LoadSpinner.start("Loading model", io)) {
                model = Models.loadSpeech(files.model(), arena, files.companions());
            } catch (IOException | IllegalArgumentException | UnsupportedOperationException e) {
                throw Main.failure("cannot load speech model '" + files.model() + "'", e);
            }
            execute(model, text, options, io, PLAYER);
            return 0;
        } finally {
            Arenas.close(arena);
        }
    }

    static void execute(
            SpeechSynthesisModel<?, ?, ?> model,
            String text,
            Options options,
            Main.IO io,
            Playback player)
            throws IOException {
        SpeechOptions speech =
                options.speech.speed == null
                        ? SpeechOptions.NONE
                        : SpeechOptions.speed(options.speech.speed);
        long start = System.nanoTime();
        if (options.stream) {
            try {
                player.stream(
                        model,
                        text,
                        speech,
                        () ->
                                io.err()
                                        .printf(
                                                "first audio after %.2f s%n",
                                                (System.nanoTime() - start) / 1e9));
            } catch (IOException e) {
                throw Main.failure("cannot stream speech", e);
            }
            return;
        }
        Media.Audio audio = model.speak(text, speech);
        double elapsed = (System.nanoTime() - start) / 1e9;
        if (options.speech.play) {
            try {
                player.play(audio);
            } catch (IOException e) {
                throw Main.failure("cannot play speech", e);
            }
            return;
        }
        byte[] wav = AudioCodec.wav(audio);
        Path output = options.speech.output;
        if (output.toString().equals("-")) {
            io.out().write(wav);
            io.out().flush();
            if (io.out().checkError()) throw new IOException("cannot write WAV to stdout");
        } else {
            try {
                Files.write(output, wav);
            } catch (IOException e) {
                throw Main.failure(
                        "cannot write WAV to '"
                                + output
                                + "'; choose a writable location with --output",
                        e);
            }
        }
        double seconds = audio.pcm().length / (double) audio.channels() / audio.sampleRate();
        io.err()
                .printf(
                        "wrote %s: %.2f s of audio in %.2f s (%.1fx realtime)%n",
                        output, seconds, elapsed, seconds / elapsed);
    }

    static void printHelp(PrintStream out) {
        out.println(
                """
                jinfer speak - turn text into speech
                Usage: jinfer [model options] speak [options] <text|->
                Examples:
                  jinfer speak -m inflect.gguf "Hello world."
                  jinfer speak -m kokoro.gguf --with voice=af_heart.gguf --output hello.wav "Hello."

                  --play                     synthesize fully, then play (default)
                  --stream                   play clips during synthesis
                  -o, --output <file|->      write WAV instead of playing; '-' writes stdout
                  --speed <number>           positive speaking-rate multiplier; default: model's rate

                Explicit output modes are mutually exclusive. '-' reads UTF-8 text to EOF;
                --stream controls audio playback, not incremental text input.
                Kokoro requires --with voice=<path|ref>; Inflect2 accepts --with lexicon=<path|ref>.
                Playback: macOS afplay; Windows PowerShell SoundPlayer; Linux aplay/ffplay.
                Legacy --speak without an output mode still writes output.wav.
                """);
        Options.modelHelp(out);
    }
}
