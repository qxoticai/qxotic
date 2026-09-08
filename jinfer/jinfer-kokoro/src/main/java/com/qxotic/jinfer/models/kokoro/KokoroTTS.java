package com.qxotic.jinfer.models.kokoro;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.memory.MemoryArena;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

/** Raw-text speech synthesis over Kokoro v1.0 and one load-time voice. */
public final class KokoroTTS
        implements SpeechSynthesisModel<Kokoro.Configuration, Kokoro.Weights, Kokoro.State> {

    private static final int CHUNK_LIMIT = 200;
    private static final double MIN_SPEED = 0.5, MAX_SPEED = 2.0;

    private final Kokoro model;
    private final KokoroPhonemizer phonemizer;

    private KokoroTTS(Kokoro model, KokoroPhonemizer phonemizer) {
        this.model = model;
        this.phonemizer = phonemizer;
    }

    public static KokoroTTS load(Path model, Path voice, Arena arena) throws IOException {
        KokoroPhonemizer phonemizer = requirePhonemizer();
        return new KokoroTTS(Kokoro.load(model, voice, arena), phonemizer);
    }

    public static KokoroTTS load(FileChannel channel, GGUF gguf, Path voice, Arena arena)
            throws IOException {
        return load(channel, gguf, 0, voice, arena);
    }

    public static KokoroTTS load(
            FileChannel channel, GGUF gguf, long baseOffset, Path voice, Arena arena)
            throws IOException {
        KokoroPhonemizer phonemizer = requirePhonemizer();
        return new KokoroTTS(Kokoro.load(channel, gguf, baseOffset, voice, arena), phonemizer);
    }

    private static KokoroPhonemizer requirePhonemizer() throws IOException {
        KokoroPhonemizer phonemizer = KokoroPhonemizer.tryCreate();
        if (phonemizer == null)
            throw new IOException("Kokoro requires espeak-ng or espeak on PATH for phonemization");
        return phonemizer;
    }

    @Override
    public Kokoro.Configuration configuration() {
        return model.configuration();
    }

    @Override
    public Kokoro.Weights weights() {
        return model.weights();
    }

    @Override
    public Kokoro.State newState() {
        return model.newState();
    }

    @Override
    public Kokoro.State newState(MemoryArena<MemorySegment> arena) {
        return model.newState(arena);
    }

    @Override
    public int sampleRate() {
        return configuration().sampleRate();
    }

    @Override
    public void speak(
            Kokoro.State state, String text, SpeechOptions options, Predicate<Media.Audio> sink) {
        if (text == null || text.isBlank()) throw new IllegalArgumentException("text is empty");
        double speed = speed(options);
        state.exclusively(
                () -> {
                    try {
                        int chunk = 0;
                        var parts = new ArrayDeque<>(chunks(text));
                        while (!parts.isEmpty()) {
                            String part = parts.removeFirst();
                            int[] phonemes = model.symbols().toRaw(phonemizer.phonemize(part));
                            if (phonemes.length == 0) continue;
                            if (phonemes.length > Kokoro.MAX_PHONEMES) {
                                if (part.codePointCount(0, part.length()) < 2)
                                    throw new IllegalArgumentException(
                                            "phonemized chunk exceeds "
                                                    + Kokoro.MAX_PHONEMES
                                                    + " symbols");
                                int split = splitAtWord(part, part.length() / 2, 1);
                                parts.addFirst(part.substring(split).trim());
                                parts.addFirst(part.substring(0, split).trim());
                                continue;
                            }
                            float[] pcm = model.synthesize(state, phonemes, speed, chunk++);
                            if (!sink.test(new Media.Audio(pcm, configuration().sampleRate(), 1)))
                                return;
                        }
                        if (chunk == 0)
                            throw new IllegalArgumentException(
                                    "text produced no supported phonemes");
                    } catch (IOException e) {
                        throw new UncheckedIOException(e);
                    }
                });
    }

    static List<String> chunks(String text) {
        String normalized = text.replaceAll("(?U)\\s+", " ").trim();
        List<String> chunks = new ArrayList<>();
        for (String sentence : normalized.split("(?<=[.!?;:])\\s+")) {
            String rest = sentence.trim();
            while (rest.length() > CHUNK_LIMIT) {
                int split = splitAtWord(rest, CHUNK_LIMIT, CHUNK_LIMIT / 2);
                chunks.add(rest.substring(0, split).trim());
                rest = rest.substring(split).trim();
            }
            if (!rest.isEmpty()) chunks.add(rest);
        }
        return chunks;
    }

    private static int splitAtWord(String text, int preferred, int earliest) {
        int split = text.lastIndexOf(' ', preferred);
        if (split < earliest) split = preferred;
        if (Character.isHighSurrogate(text.charAt(split - 1))
                && Character.isLowSurrogate(text.charAt(split))) split--;
        return split;
    }

    private static double speed(SpeechOptions options) {
        Double speed = options == null ? null : options.speed();
        if (speed == null) return 1;
        if (speed < MIN_SPEED || speed > MAX_SPEED)
            throw new IllegalArgumentException(
                    "speed must be in [" + MIN_SPEED + ", " + MAX_SPEED + "]: " + speed);
        return speed;
    }
}
