package com.qxotic.jinfer.models.kokoro;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Phonemizer;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.SpeechSynthesisModel;
import com.qxotic.jinfer.codecs.Espeak;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.memory.MemoryArena;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Predicate;

/** Speech synthesis over Kokoro v1.0 and one load-time voice. */
public final class KokoroTTS
        implements SpeechSynthesisModel<Kokoro.Configuration, Kokoro.Weights, Kokoro.State> {

    private static final int CHUNK_LIMIT = 200;

    private final Kokoro model;
    private final Phonemizer phonemizer;

    private KokoroTTS(Kokoro model, Phonemizer phonemizer) {
        this.model = model;
        this.phonemizer = phonemizer;
    }

    public static KokoroTTS load(Path model, Path voice, Arena arena) throws IOException {
        return wrap(Kokoro.load(model, voice, arena));
    }

    public static KokoroTTS load(FileChannel channel, GGUF gguf, Path voice, Arena arena)
            throws IOException {
        return load(channel, gguf, 0, voice, arena);
    }

    public static KokoroTTS load(
            FileChannel channel, GGUF gguf, long baseOffset, Path voice, Arena arena)
            throws IOException {
        return wrap(Kokoro.load(channel, gguf, baseOffset, voice, arena));
    }

    /**
     * The front end: espeak in the voice's language, rewritten into misaki's dialect (the one
     * Kokoro was trained on), through the symbol table the GGUF declares. Kokoro has no lexicon, so
     * espeak is required rather than asked for.
     */
    private static KokoroTTS wrap(Kokoro kokoro) throws IOException {
        Espeak espeak = Espeak.find().orElse(null);
        if (espeak == null)
            throw new IOException("Kokoro requires espeak-ng or espeak on PATH for phonemization");
        String language = kokoro.language();
        Misaki dialect = Misaki.forLanguage(language);
        return new KokoroTTS(
                kokoro,
                Phonemizer.ipa(
                        List.of(kokoro.configuration().tokens()),
                        run -> dialect.apply(espeak.tiedIpa(run, language))));
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
    public Phonemizer phonemizer() {
        return phonemizer;
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
    public Media.Audio synthesize(Kokoro.State state, int[] phonemes, SpeechOptions options) {
        Double speed = options.speed(); // Kokoro bounds it itself
        float[] pcm = model.synthesize(state, phonemes, speed == null ? 1 : speed, 0);
        return new Media.Audio(pcm, sampleRate(), 1);
    }

    @Override
    public void speak(
            Kokoro.State state, String text, SpeechOptions options, Predicate<Media.Audio> sink) {
        if (text.isBlank()) throw new IllegalArgumentException("text is blank");
        state.exclusively(
                () -> {
                    boolean spoke = false;
                    var parts = new ArrayDeque<>(chunks(text));
                    while (!parts.isEmpty()) {
                        String part = parts.removeFirst();
                        int[] phonemes = phonemizer.phonemize(part);
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
                        if (!sink.test(synthesize(state, phonemes, options))) return;
                        spoke = true;
                    }
                    if (!spoke)
                        throw new IllegalArgumentException("text produced no supported phonemes");
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
}
