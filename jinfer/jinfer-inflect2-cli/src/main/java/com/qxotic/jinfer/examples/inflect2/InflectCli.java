// Command line for Inflect2: synthesize text to a WAV file, or speak it aloud.
//
//   inflect model.gguf --text "Hello world." --output hello.wav
//   inflect --model z://default.gguf --play --speed 1.1
package com.qxotic.jinfer.examples.inflect2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.models.inflect2.Inflect2;
import com.qxotic.jinfer.models.inflect2.InflectTTS;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.Channels;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicBoolean;

public final class InflectCli {
    private InflectCli() {}

    private static final String SELF_ARCHIVE = "z://";
    private static final String DEFAULT_ENTRY = "default.gguf";

    public static void main(String[] args) {
        try {
            run(Options.parse(args));
        } catch (IllegalArgumentException badCommandLine) {
            System.err.println("inflect: " + badCommandLine.getMessage());
            Options.usage(System.err);
            System.exit(2);
        } catch (Player.Failed playerQuit) {
            // The player's status becomes ours: a script can tell a refused format from a
            // missing device without parsing anything we print.
            System.err.println("inflect: " + playerQuit.getMessage());
            System.exit(playerQuit.status);
        } catch (IOException e) {
            // A player that will not start and a path that will not open are both the user's to
            // fix, and the message says how. A stack trace through the synthesis loop does not.
            System.err.println("inflect: " + e.getMessage());
            System.exit(1);
        }
    }

    /** The whole program, once the command line is understood: load a voice, then use it. */
    private static void run(Options options) throws IOException {
        if (options.help()) {
            Options.usage(System.out);
        } else if (options.list()) {
            list(options.model());
        } else {
            InflectTTS tts = tuned(open(options.model()), options);
            // One state for the whole run: minting one per utterance repays every sizing
            // allocation, and closes a shared arena, which is a JVM-wide handshake.
            try (Inflect2.State state = tts.newState()) {
                if (options.play()) play(tts, state, options);
                else write(tts, state, options);
            }
        }
    }

    /** The knobs that are this model's own, applied as a re-wrap over the same weights. */
    private static InflectTTS tuned(InflectTTS tts, Options options) {
        InflectTTS tuned = tts.variation(options.variation()).seed(options.seed());
        return options.overrides().isEmpty() ? tuned : tuned.wordOverrides(options.overrides());
    }

    /** Resolve a model: a {@code z://} entry, a file path, or the embedded default. */
    private static InflectTTS open(String model) throws IOException {
        // a CLI's weights live exactly as long as the process: the global arena owns them
        Arena weights = Arena.global();
        if (model == null) return loadSelfArchive(DEFAULT_ENTRY, weights);
        if (model.startsWith(SELF_ARCHIVE))
            return loadSelfArchive(model.substring(SELF_ARCHIVE.length()), weights);
        Path path = Path.of(model);
        if (!Files.isReadable(path)) throw new IOException("cannot read model: " + path);
        return InflectTTS.load(path, weights);
    }

    /** Map a GGUF straight from a STORED entry in the running executable's ZIP overlay. */
    private static InflectTTS loadSelfArchive(String name, Arena arena) throws IOException {
        try (SelfArchive archive = SelfArchive.open()) {
            SelfArchive.Entry entry = archive.entry(name);
            // The header is small (< 64 KB even for Inflect2's 302 tensors).
            byte[] header = archive.readAt(entry.offset(), (int) Math.min(entry.size(), 1 << 16));
            GGUF gguf = GGUF.read(Channels.newChannel(new ByteArrayInputStream(header)));
            return InflectTTS.load(archive.channel(), gguf, entry.offset(), arena);
        }
    }

    /** Synthesize the whole utterance into {@code wav}, saying how fast the model ran. */
    private static Path render(InflectTTS tts, Inflect2.State state, Options options, Path wav)
            throws IOException {
        long from = System.nanoTime();
        var audio = tts.speak(state, options.text(), SpeechOptions.speed(options.speed()));
        double elapsed = (System.nanoTime() - from) / 1e9;
        double seconds = audio.pcm().length / (double) audio.sampleRate();
        AudioIO.writeWav(audio.pcm(), audio.sampleRate(), wav);
        System.out.printf(
                "%.2f s of audio in %.2f s (%.1fx realtime)%n",
                seconds, elapsed, seconds / elapsed);
        return wav;
    }

    private static void write(InflectTTS tts, Inflect2.State state, Options options)
            throws IOException {
        System.out.println(
                "wrote " + render(tts, state, options, options.output()).toAbsolutePath());
    }

    /**
     * Speak it aloud: streamed to the player clip by clip where that is possible, and through a
     * temporary WAV where it is not. {@link Player} says which players are which, and why.
     */
    private static void play(InflectTTS tts, Inflect2.State state, Options options)
            throws IOException {
        int rate = tts.sampleRate();
        Player speaker = Player.streaming(rate);
        if (speaker == null) {
            // Nothing here reads a pipe: finish the utterance and hand over the file.
            Path wav = Files.createTempFile("inflect", ".wav");
            try {
                Player.play(render(tts, state, options, wav));
            } finally {
                Files.deleteIfExists(wav);
            }
            return;
        }
        long from = System.nanoTime();
        var announced = new AtomicBoolean();
        try (speaker) {
            // A moment of silence, so the player has something to chew on while the first clip is
            // still being synthesized.
            speaker.offer(AudioCodec.pcm16(new Media.Audio(new float[rate / 32], rate, 1)));
            // Headerless PCM, so the clips concatenate; the player was told the format up front.
            // The pipe IS the overlap: offer() blocks once it is full, so the next clip is
            // synthesized while the player drains this one.
            tts.speak(
                    state,
                    options.text(),
                    SpeechOptions.speed(options.speed()),
                    clip -> {
                        if (!speaker.offer(AudioCodec.pcm16(clip))) return false;
                        // Latency is what streaming is about, and the only honest number here:
                        // past the first clip the pipe paces us, so a rate would be measuring the
                        // speaker. --output measures how fast the model runs.
                        if (announced.compareAndSet(false, true))
                            System.out.printf(
                                    "first audio after %.2f s%n", (System.nanoTime() - from) / 1e9);
                        return true;
                    });
        }
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
        var config = inflect.configuration();
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
}
