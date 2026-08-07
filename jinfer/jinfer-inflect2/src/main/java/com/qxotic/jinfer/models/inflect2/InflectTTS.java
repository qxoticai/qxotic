// Text-to-speech over Inflect2: normalize, phonemize, synthesize sentence by sentence.
//
//   InflectTTS tts = InflectTTS.load(Path.of("model.gguf"));
//   try (Inflect2.State state = tts.newState()) {
//       Media.Audio audio = tts.speak(state, "Hello world.", SpeechOptions.NONE);
//   }
//
// Long text is split into sentence-sized chunks, each synthesized separately and faded at its
// edges, with a punctuation-dependent pause between them (as in the reference implementation).
package com.qxotic.jinfer.models.inflect2;

import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.SpeechModel;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.models.inflect2.frontend.Phonemizer;
import com.qxotic.jinfer.models.inflect2.frontend.TextNormalizer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.function.Predicate;

public final class InflectTTS
        implements SpeechModel<Inflect2.Configuration, Inflect2.Weights, Inflect2.State> {

    /** Longest chunk handed to the model; longer sentences are split at a comma, then a space. */
    private static final int CHUNK_LIMIT = 280;

    /** Edge fade applied to every chunk, to keep the joins from clicking. */
    private static final double FADE_MILLIS = 5;

    /** This family's defaults, for the knobs a caller leaves unset. */
    private static final double DEFAULT_SPEED = 1.0, DEFAULT_VARIATION = 0.667;

    /** A rate multiplies every predicted duration, so it is also a cost multiplier: bound it. */
    private static final double MIN_SPEED = 0.5, MAX_SPEED = 2.0;

    private final Inflect2 model;
    private final Phonemizer phonemizer;
    private final Map<String, String> wordOverrides;
    private final double variation;
    private final long seed;

    private InflectTTS(
            Inflect2 model,
            Phonemizer phonemizer,
            Map<String, String> wordOverrides,
            double variation,
            long seed) {
        this.model = model;
        this.phonemizer = phonemizer;
        this.wordOverrides = wordOverrides;
        this.variation = variation;
        this.seed = seed;
    }

    // ── loading ───────────────────────────────────────────────────────────

    /**
     * Weights map into an {@code ofAuto} arena. Correct here and only here: this port MAPS its
     * tensors READ_ONLY and materializes nothing at load, so the pages are kernel-reclaimable and
     * GC scheduling is irrelevant. There is nothing to close.
     */
    public static InflectTTS load(Path gguf) throws IOException {
        return wrap(Inflect2.load(gguf), gguf, null);
    }

    /**
     * Weights map into {@code arena}, whose owner is whoever provided it — see {@link
     * Inflect2#load(Path, Arena)} for the lifetime rules. Synthesis states are separate and own
     * their own scratch unless you say otherwise.
     */
    public static InflectTTS load(Path gguf, Arena arena) throws IOException {
        return wrap(Inflect2.load(gguf, arena), gguf, null);
    }

    /**
     * As {@link #load(Path, Arena)} with an explicit pronunciation lexicon: a {@code .bin} file, or
     * a directory holding one per language. An unreadable {@code lexicon} THROWS — naming a file
     * and silently getting the fallback is the lie the ladder must not tell.
     */
    public static InflectTTS load(Path gguf, Arena arena, Path lexicon) throws IOException {
        if (lexicon == null) throw new IllegalArgumentException("null lexicon");
        return wrap(Inflect2.load(gguf, arena), gguf, lexicon);
    }

    /**
     * As {@link #load(Path, Arena)} but reusing an already-parsed {@code gguf} - the arch-dispatch
     * entry ({@code Models.loadSpeech}). {@code path} is where that GGUF lives, so the lexicon
     * beside it is still found.
     */
    public static InflectTTS load(
            java.nio.channels.FileChannel channel,
            com.qxotic.format.gguf.GGUF gguf,
            Path path,
            Arena arena)
            throws IOException {
        return wrap(Inflect2.load(channel, gguf, arena), path, null);
    }

    /**
     * As {@link #load(FileChannel, GGUF, Path, Arena)} with an explicit pronunciation lexicon,
     * which REPLACES the discovery ladder rather than joining it: naming a file and silently
     * falling back to another would be the same lie as ignoring it. Unreadable throws.
     */
    public static InflectTTS load(
            java.nio.channels.FileChannel channel,
            com.qxotic.format.gguf.GGUF gguf,
            Path path,
            Arena arena,
            Path lexicon)
            throws IOException {
        if (lexicon == null) throw new IllegalArgumentException("null lexicon");
        return wrap(Inflect2.load(channel, gguf, arena), path, lexicon);
    }

    /** Load from a ZIP overlay appended to the running executable, e.g. {@code "default.gguf"}. */
    public static InflectTTS loadSelfArchive(String entryName) throws IOException {
        return wrap(Inflect2.loadSelfArchive(entryName), null, null);
    }

    /** As {@link #loadSelfArchive(String)}, with the weights mapped into {@code arena}. */
    public static InflectTTS loadSelfArchive(String entryName, Arena arena) throws IOException {
        return wrap(Inflect2.loadSelfArchive(entryName, arena), null, null);
    }

    private static InflectTTS wrap(Inflect2 model, Path gguf, Path lexicon) throws IOException {
        return new InflectTTS(model, frontend(gguf, lexicon), Map.of(), DEFAULT_VARIATION, 0);
    }

    /**
     * The phonemizer, in one place and one order: the lexicon you named, the one beside the GGUF,
     * the one on the classpath, then espeak-ng, then nothing works and we say so.
     *
     * <p>The classpath rung exists for the native image, which has no directory to sit beside. The
     * lexicon is NOT a checked-in resource - it is 1.5 MB of downloaded fixture, copied into the
     * image at build time by the {@code native} profile from wherever the models were downloaded.
     * An image built without it falls back to espeak-ng, which is a working image.
     *
     * <p>The two are alternatives, not layers. A lexicon is a hash lookup and knows only what it
     * was built with, leaving the rest unspoken (it says so on stderr — {@link #wordOverrides(Map)}
     * is the fix for a handful of terms). espeak has a letter-to-sound model and pronounces
     * anything, at ~50x realtime against the lexicon's ~54x.
     */
    private static Phonemizer frontend(Path gguf, Path lexicon) throws IOException {
        if (lexicon != null) return Phonemizer.lexicon(lexicon);
        if (gguf != null) {
            Path beside = gguf.resolveSibling("lexicon.bin");
            if (Files.isReadable(beside)) return Phonemizer.lexicon(beside);
        }
        Phonemizer bundled = Phonemizer.bundledLexicon();
        if (bundled != null) return bundled;
        Phonemizer espeak = Phonemizer.espeak();
        if (espeak != null) {
            System.getLogger("jinfer.inflect2")
                    .log(
                            System.Logger.Level.WARNING,
                            "no lexicon found, using espeak-ng: one subprocess per"
                                    + " punctuation-free run, and it must stay installed. Ship a"
                                    + " lexicon.bin beside the GGUF to avoid it.");
            return espeak;
        }
        throw new IOException(
                "no phonemizer: pass a lexicon to InflectTTS.load(gguf, arena, lexicon), put"
                    + " lexicon.bin beside the model or on the classpath, or install espeak-ng");
    }

    // ── tuning: a re-wrap over the SAME weights, so no reload and no arena ─

    /** VITS latent noise scale, 0..1, default 0.667. Lower is flatter and more repeatable. */
    public InflectTTS variation(double variation) {
        if (!(variation >= 0 && variation <= 1))
            throw new IllegalArgumentException("variation must be in [0, 1]: " + variation);
        return new InflectTTS(model, phonemizer, wordOverrides, variation, seed);
    }

    /** Every {@link #speak} starts here: same state, same text, same waveform. */
    public InflectTTS seed(long seed) {
        return new InflectTTS(model, phonemizer, wordOverrides, variation, seed);
    }

    /**
     * Pronunciation overrides — terms rewritten to readable English before phonemization, e.g.
     * {@code "PyTorch" → "pie torch"}. User entries take priority over the built-in table, and this
     * is the supported fix for words a lexicon does not know.
     */
    public InflectTTS wordOverrides(Map<String, String> overrides) {
        return new InflectTTS(model, phonemizer, Map.copyOf(overrides), variation, seed);
    }

    // ── SpeechModel ───────────────────────────────────────────────────────

    public Inflect2 model() {
        return model;
    }

    @Override
    public Inflect2.Configuration config() {
        return model.config();
    }

    @Override
    public Inflect2.Weights weights() {
        return model.weights();
    }

    @Override
    public Inflect2.State newState(Arena arena, boolean adopt) {
        return model.newState(arena, adopt);
    }

    /** ponytail: kept as a one-liner over config().sampleRate(), used four times in the CLI. */
    public int sampleRate() {
        return model.sampleRate();
    }

    @Override
    public void speak(
            Inflect2.State state, String text, SpeechOptions options, Predicate<Media.Audio> sink) {
        double speed = speed(options);
        List<String> chunks = split(text);
        // Claim the state for the WHOLE utterance, not per chunk: a close arriving between chunks
        // would otherwise free the arena mid-utterance. Reentrant, so the per-chunk synthesize
        // nests inside this one.
        state.enter();
        try {
            for (int i = 0; i < chunks.size(); i++) {
                if (i > 0 && !sink.test(silence(pauseSamples(chunks.get(i - 1))))) return;
                float[] chunk = synthesizeChunk(state, chunks.get(i), speed, seed + i);
                if (!sink.test(new Media.Audio(chunk, sampleRate(), 1))) return;
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } finally {
            state.exit();
        }
    }

    /** The caller's rate, bounded. A speed outside the range is refused, never clamped. */
    private static double speed(SpeechOptions options) {
        Double speed = options == null ? null : options.speed();
        if (speed == null) return DEFAULT_SPEED;
        if (!(speed >= MIN_SPEED && speed <= MAX_SPEED))
            throw new IllegalArgumentException(
                    "speed must be in [" + MIN_SPEED + ", " + MAX_SPEED + "]: " + speed);
        return speed;
    }

    private Media.Audio silence(int samples) {
        return new Media.Audio(new float[samples], sampleRate(), 1);
    }

    private float[] synthesizeChunk(Inflect2.State state, String chunk, double speed, long seed)
            throws IOException {
        String normalized = TextNormalizer.normalize(chunk, wordOverrides);
        int[] tokens =
                phonemizer != null
                        ? phonemizer.phonemize(normalized)
                        : Symbols.toTokens(normalized);
        if (tokens.length == 0) return new float[0];
        // The model takes a length scale, which is the reciprocal of a speaking rate.
        Media.Audio audio =
                model.synthesize(state, tokens, (float) (1.0 / speed), (float) variation, seed);
        return clamp(fadeEdges(audio.pcm()));
    }

    // ── chunking ──────────────────────────────────────────────────────────

    /** Sentence-sized chunks: split on sentence punctuation, then break anything still too long. */
    static List<String> split(String text) {
        String normalized = text.replaceAll("\\s+", " ").trim();
        List<String> chunks = new ArrayList<>();
        for (String sentence : normalized.split("(?<=[.!?;:])\\s+")) {
            String rest = sentence.trim();
            while (rest.length() > CHUNK_LIMIT) {
                int at = breakPoint(rest);
                chunks.add(rest.substring(0, at).trim());
                rest = rest.substring(at).trim();
            }
            if (!rest.isEmpty()) chunks.add(rest);
        }
        return chunks.isEmpty() ? List.of(normalized) : chunks;
    }

    /** Where to cut an over-long sentence: last clause break, else last space, else hard cut. */
    private static int breakPoint(String sentence) {
        int minimum = CHUNK_LIMIT / 2;
        for (char mark : new char[] {',', ';', ':'}) {
            int at = sentence.lastIndexOf(mark, CHUNK_LIMIT);
            if (at >= minimum) return at + 1;
        }
        int space = sentence.lastIndexOf(' ', CHUNK_LIMIT);
        return space >= minimum ? space : CHUNK_LIMIT;
    }

    /** Pause after a chunk, by its final punctuation — a question rests longer than a comma. */
    private int pauseSamples(String chunk) {
        String trimmed = chunk.stripTrailing();
        double seconds =
                trimmed.isEmpty()
                        ? 0.08
                        : switch (trimmed.charAt(trimmed.length() - 1)) {
                            case '?' -> 0.28;
                            case '!' -> 0.24;
                            case '.' -> 0.22;
                            case ';' -> 0.16;
                            case ':' -> 0.13;
                            case ',' -> 0.09;
                            default -> 0.08;
                        };
        return (int) Math.round(sampleRate() * seconds);
    }

    /** Ramp the first and last few milliseconds of a chunk in and out. */
    private float[] fadeEdges(float[] pcm) {
        int frames = (int) Math.min(Math.round(sampleRate() * FADE_MILLIS / 1000), pcm.length / 2);
        for (int i = 0; i < frames; i++) {
            float ramp = (float) i / frames;
            pcm[i] *= ramp;
            pcm[pcm.length - 1 - i] *= ramp;
        }
        return pcm;
    }

    /** Per CLIP, not per join: a streamed clip and a concatenated one must be the same samples. */
    private static float[] clamp(float[] pcm) {
        for (int i = 0; i < pcm.length; i++) pcm[i] = Math.clamp(pcm[i], -1f, 1f);
        return pcm;
    }
}
