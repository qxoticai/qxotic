// Text-to-speech over Inflect2: normalize, phonemize, synthesize sentence by sentence.
//
//   InflectTTS tts = InflectTTS.load(Path.of("model.gguf"));
//   Media.Audio audio = tts.synthesize("Hello world.", 1.0, 0.667, 42);
//
// Long text is split into sentence-sized chunks, each synthesized separately and faded at its
// edges, with a punctuation-dependent pause between them (as in the reference implementation).
package com.qxotic.jinfer.models.inflect2;

import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.models.inflect2.frontend.Phonemizer;
import com.qxotic.jinfer.models.inflect2.frontend.TextNormalizer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

public final class InflectTTS {

    /** Longest chunk handed to the model; longer sentences are split at a comma, then a space. */
    private static final int CHUNK_LIMIT = 280;

    /** Edge fade applied to every chunk, to keep the joins from clicking. */
    private static final double FADE_MILLIS = 5;

    private final Inflect2 model;
    private final Phonemizer phonemizer;
    private volatile Map<String, String> wordOverrides = Map.of();

    private InflectTTS(Inflect2 model) {
        this.model = model;
        this.phonemizer = Phonemizer.tryCreate();
    }

    public static InflectTTS load(Path ggufPath) throws IOException {
        return new InflectTTS(Inflect2.load(ggufPath));
    }

    /** Load from a ZIP overlay appended to the running executable, e.g. {@code "default.gguf"}. */
    public static InflectTTS loadSelfArchive(String entryName) throws IOException {
        return new InflectTTS(Inflect2.loadSelfArchive(entryName));
    }

    public Inflect2 model() {
        return model;
    }

    public int sampleRate() {
        return model.sampleRate();
    }

    /**
     * Pronunciation overrides — terms rewritten to readable English before phonemization, e.g.
     * {@code "PyTorch" → "pie torch"}. Applied from the next call on; user entries take priority
     * over the built-in table.
     */
    public void setWordOverrides(Map<String, String> overrides) {
        this.wordOverrides = Map.copyOf(overrides);
    }

    /** The whole text as one waveform: chunks, inter-sentence pauses, clipped to [-1,1]. */
    public Media.Audio synthesize(String text, double speed, double variation, long seed)
            throws IOException {
        List<String> chunks = split(text);
        // One scratch state for the whole text: the second chunk onward allocates nothing.
        Inflect2.State state = model.newState();
        List<float[]> pieces = new ArrayList<>(chunks.size() * 2);
        int total = 0;
        for (int i = 0; i < chunks.size(); i++) {
            if (i > 0) {
                float[] pause = new float[pauseSamples(chunks.get(i - 1))];
                pieces.add(pause);
                total += pause.length;
            }
            float[] chunk = synthesizeChunk(state, chunks.get(i), speed, variation, seed + i);
            pieces.add(chunk);
            total += chunk.length;
        }
        float[] pcm = new float[total];
        int offset = 0;
        for (float[] piece : pieces) {
            System.arraycopy(piece, 0, pcm, offset, piece.length);
            offset += piece.length;
        }
        for (int i = 0; i < pcm.length; i++) pcm[i] = Math.clamp(pcm[i], -1f, 1f);
        return new Media.Audio(pcm, sampleRate(), 1);
    }

    /**
     * One waveform chunk per sentence, synthesized on demand as the stream is consumed — so a
     * consumer can start playing or encoding before the whole text is done. Pauses are the
     * consumer's business here; {@link #synthesize} inserts them.
     */
    public Stream<float[]> stream(String text, double speed, double variation, long seed) {
        List<String> chunks = split(text);
        // The stream owns one state, so a consumer walking it allocates nothing per chunk. A stream
        // is single-consumer for that reason; take two streams to synthesize on two threads.
        Inflect2.State state = model.newState();
        return java.util.stream.IntStream.range(0, chunks.size())
                .mapToObj(
                        i -> {
                            try {
                                return synthesizeChunk(
                                        state, chunks.get(i), speed, variation, seed + i);
                            } catch (IOException e) {
                                throw new UncheckedIOException(e);
                            }
                        });
    }

    private float[] synthesizeChunk(
            Inflect2.State state, String chunk, double speed, double variation, long seed)
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
        return fadeEdges(audio.pcm());
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
}
