// Command line for Inflect2: synthesize text to a WAV file, or pipe it to a system player.
//
//   inflect model.gguf --text "Hello world." --output hello.wav
//   inflect --model z://default.gguf --play --speed 1.1
package com.qxotic.jinfer.models.inflect2;

import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.media.AudioCodec;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

public final class InflectCli {
    private InflectCli() {}

    private static final String SELF_ARCHIVE = "z://";
    private static final String DEFAULT_ENTRY = "default.gguf";

    public static void main(String[] args) throws IOException {
        Options options;
        try {
            options = Options.parse(args);
        } catch (IllegalArgumentException e) {
            System.err.println("inflect: " + e.getMessage());
            usage(System.err);
            System.exit(2);
            return;
        }
        if (options.help) {
            usage(System.out);
            return;
        }
        if (options.list) {
            list(options.model);
            return;
        }

        InflectTTS tts = tuned(open(options.model), options);
        // one state for the whole run: minting one per utterance repays every sizing allocation
        // and closes a shared arena, which is a JVM-wide handshake
        try (Inflect2.State state = tts.newState()) {
            if (options.play) play(tts, state, options);
            else write(tts, state, options);
        }
    }

    /** The knobs that are this model's own, applied as a re-wrap over the same weights. */
    private static InflectTTS tuned(InflectTTS tts, Options options) {
        InflectTTS tuned = tts.variation(options.variation).seed(options.seed);
        return options.overrides.isEmpty() ? tuned : tuned.wordOverrides(options.overrides);
    }

    /** Resolve a model: a {@code z://} entry, a file path, or the embedded default. */
    private static InflectTTS open(String model) throws IOException {
        if (model == null) return InflectTTS.loadSelfArchive(DEFAULT_ENTRY);
        if (model.startsWith(SELF_ARCHIVE))
            return InflectTTS.loadSelfArchive(model.substring(SELF_ARCHIVE.length()));
        Path path = Path.of(model);
        if (!Files.isReadable(path)) throw new IOException("cannot read model: " + path);
        return InflectTTS.load(path);
    }

    private static void write(InflectTTS tts, Inflect2.State state, Options options)
            throws IOException {
        long start = System.nanoTime();
        var audio = tts.speak(state, options.text, delivery(options));
        double elapsed = (System.nanoTime() - start) / 1e9;
        float[] pcm = audio.pcm();
        double seconds = pcm.length / (double) audio.sampleRate();
        float peak = 0;
        double energy = 0;
        for (float sample : pcm) {
            peak = Math.max(peak, Math.abs(sample));
            energy += (double) sample * sample;
        }
        AudioIO.writeWav(pcm, audio.sampleRate(), options.output);
        System.out.printf(
                "%.2f s of audio in %.2f s (%.1f× realtime), peak %.3f, rms %.4f%n",
                seconds, elapsed, seconds / elapsed, peak, Math.sqrt(energy / pcm.length));
        System.out.println("wrote " + options.output.toAbsolutePath());
    }

    /**
     * Stream to a system player. The player is a separate process, so writing into its pipe is
     * already the overlap: synthesis of the next chunk proceeds while the player drains this one.
     */
    private static SpeechOptions delivery(Options options) {
        return SpeechOptions.speed(options.speed);
    }

    private static void play(InflectTTS tts, Inflect2.State state, Options options)
            throws IOException {
        Process player =
                new ProcessBuilder(playerCommand(tts.sampleRate()))
                        .redirectError(ProcessBuilder.Redirect.DISCARD)
                        .start();
        long start = System.nanoTime();
        int[] samples = {0};
        boolean[] first = {true};
        try (OutputStream pipe = player.getOutputStream()) {
            // A short lead-in keeps the player from starving before the first chunk lands.
            pipe.write(
                    AudioCodec.pcm16(
                            new Media.Audio(
                                    new float[tts.sampleRate() / 32], tts.sampleRate(), 1)));
            // Headerless PCM, so the pieces concatenate; the player was told the format on its
            // command line. A pipe that closes (the user quit the player) cancels the synthesis
            // instead of filling a dead buffer.
            tts.speak(
                    state,
                    options.text,
                    delivery(options),
                    clip -> {
                        if (first[0]) {
                            System.out.printf(
                                    "first audio after %.2f s%n",
                                    (System.nanoTime() - start) / 1e9);
                            first[0] = false;
                        }
                        samples[0] += clip.pcm().length;
                        try {
                            pipe.write(AudioCodec.pcm16(clip));
                            pipe.flush();
                            return true;
                        } catch (IOException e) {
                            return false; // the player quit: stop synthesizing for a dead pipe
                        }
                    });
        }
        double elapsed = (System.nanoTime() - start) / 1e9;
        double seconds = samples[0] / (double) tts.sampleRate();
        System.out.printf(
                "%.2f s of audio in %.2f s (%.1f× realtime)%n",
                seconds, elapsed, seconds / elapsed);
        try {
            player.waitFor();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            player.destroy();
        }
    }

    /** Raw S16LE over stdin — aplay where present, else ffplay. */
    private static String[] playerCommand(int sampleRate) throws IOException {
        String rate = String.valueOf(sampleRate);
        if (onPath("aplay"))
            return new String[] {"aplay", "-f", "S16_LE", "-r", rate, "-c", "1", "-"};
        if (onPath("ffplay"))
            return new String[] {
                "ffplay", "-f", "s16le", "-ar", rate, "-ac", "1", "-nodisp", "-autoexit", "-"
            };
        throw new IOException("no audio player found — install aplay or ffplay");
    }

    private static boolean onPath(String command) {
        String path = System.getenv("PATH");
        if (path == null) return false;
        for (String directory : path.split(java.io.File.pathSeparator))
            if (Files.isExecutable(Path.of(directory, command))) return true;
        return false;
    }

    /** Models inside the executable's ZIP overlay, with the config of the selected one. */
    private static void list(String model) throws IOException {
        if (model == null || model.startsWith(SELF_ARCHIVE)) {
            try (SelfArchive archive = SelfArchive.open()) {
                System.out.printf("%-40s %10s%n", "entry", "size");
                for (SelfArchive.Entry entry : archive.entries())
                    System.out.printf("%-40s %7.1f MB%n", entry.name(), entry.size() / 1e6);
            } catch (IOException e) {
                System.err.println("no self-archive: " + e.getMessage());
            }
            if (model == null) return;
        }
        var inflect = open(model).model();
        var config = inflect.config();
        System.out.printf(
                "symbols=%d inter=%d hidden=%d filter=%d heads=%d layers=%d sampleRate=%d"
                        + " decoderWidth=%d%n",
                config.symbolCount(),
                config.interChannels(),
                config.hiddenChannels(),
                config.filterChannels(),
                config.nHeads(),
                config.nLayers(),
                config.sampleRate(),
                config.upsampleInitialChannel());
        System.out.printf(
                "tensors=%d parameters=%d%n", inflect.tensorCount(), inflect.parameterCount());
    }

    private static void usage(java.io.PrintStream to) {
        to.println(
                """
                usage: inflect [model.gguf] [options]

                  --model <path>      model file, or z://<entry> from the executable's overlay
                  --text <string>     text to speak (default: "Hello world.")
                  --output <path>     WAV file to write (default: output.wav)
                  --speed <0.5-2.0>   speaking rate (default: 1.0)
                  --variation <0-1>   latent noise scale (default: 0.667)
                  --seed <int>        random seed (default: 7)
                  --override <k> <v>  pronunciation override, repeatable
                  --play              pipe audio to aplay/ffplay instead of writing a file
                  --list              list models in the executable's overlay, and their config
                  --help              show this message\
                """);
    }

    /** Parsed command line. Values are range-checked here so failures name the flag. */
    private record Options(
            String model,
            String text,
            Path output,
            double speed,
            double variation,
            long seed,
            boolean play,
            boolean list,
            boolean help,
            Map<String, String> overrides) {

        static Options parse(String[] args) {
            String model = null, text = "Hello world.", output = "output.wav";
            double speed = 1.0, variation = 0.667;
            long seed = 7;
            boolean play = false, list = false, help = args.length == 0;
            var overrides = new LinkedHashMap<String, String>();

            for (int i = 0; i < args.length; i++) {
                String arg = args[i];
                switch (arg) {
                    case "--model" -> model = value(args, ++i, arg);
                    case "--text" -> text = value(args, ++i, arg);
                    case "--output" -> output = value(args, ++i, arg);
                    case "--speed" -> speed = number(value(args, ++i, arg), arg, 0.5, 2.0);
                    case "--variation" -> variation = number(value(args, ++i, arg), arg, 0, 1);
                    case "--seed" -> seed = Long.parseLong(value(args, ++i, arg));
                    case "--play" -> play = true;
                    case "--list" -> list = true;
                    case "--help", "-h" -> help = true;
                    case "--override" -> {
                        String key = value(args, ++i, arg);
                        overrides.put(key, value(args, ++i, arg));
                    }
                    default -> {
                        if (arg.startsWith("-"))
                            throw new IllegalArgumentException("unknown flag " + arg);
                        if (model != null)
                            throw new IllegalArgumentException("more than one model given: " + arg);
                        model = arg;
                    }
                }
            }
            return new Options(
                    model,
                    text,
                    Path.of(output),
                    speed,
                    variation,
                    seed,
                    play,
                    list,
                    help,
                    Map.copyOf(overrides));
        }

        private static String value(String[] args, int index, String flag) {
            if (index >= args.length) throw new IllegalArgumentException(flag + " needs a value");
            return args[index];
        }

        private static double number(String raw, String flag, double low, double high) {
            double value;
            try {
                value = Double.parseDouble(raw);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(flag + " expects a number, got " + raw);
            }
            if (value < low || value > high)
                throw new IllegalArgumentException(flag + " must be in [" + low + "," + high + "]");
            return value;
        }
    }
}
