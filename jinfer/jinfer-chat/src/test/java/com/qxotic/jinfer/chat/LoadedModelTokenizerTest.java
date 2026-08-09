package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertIterableEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Config;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * The cache-key contract of {@link LoadedModel#withTokenizer}: prompt-cache artifacts are keyed by
 * the seed and hold token IDS, so the seed must follow the tokenizer's BEHAVIOUR - different
 * encodings must not share a key, identical encodings must.
 */
final class LoadedModelTokenizerTest {

    @Test
    void adifferentEncodingGetsAdifferentSeed() {
        LoadedModel<?> base = loaded(StubTokenizer.encodingEvery(1));
        byte[] original = base.seed();

        byte[] same = base.withTokenizer(StubTokenizer.encodingEvery(1)).seed();
        byte[] other = base.withTokenizer(StubTokenizer.encodingEvery(2)).seed();

        assertFalse(Arrays.equals(original, same), "overriding the tokenizer re-roots the seed");
        assertFalse(
                Arrays.equals(same, other),
                "tokenizers that encode differently must not" + " share a prompt-cache key");
        assertArrayEquals(
                same,
                base.withTokenizer(StubTokenizer.encodingEvery(1)).seed(),
                "same behaviour, same key - a valid artifact must still mount");
        assertArrayEquals(original, base.seed(), "the original record is untouched");
    }

    @Test
    void everythingButTheSeedAndTokenizerSurvives() {
        LoadedModel<?> base = loaded(StubTokenizer.encodingEvery(1));
        Tokenizer replacement = StubTokenizer.encodingEvery(3);
        LoadedModel<?> derived = base.withTokenizer(replacement);

        assertSame(base.model(), derived.model());
        assertSame(replacement, derived.tokenizer());
        assertSame(base.chatTemplateSource(), derived.chatTemplateSource());
        assertArrayEquals(
                base.stopTokens().stream().mapToInt(Integer::intValue).sorted().toArray(),
                derived.stopTokens().stream().mapToInt(Integer::intValue).sorted().toArray());
    }

    @Test
    void nullIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () -> loaded(StubTokenizer.encodingEvery(1)).withTokenizer(null));
    }

    @Test
    void stopTokenOrderSurvivesTheDefensiveCopy() {
        // the order carries meaning: SpecialTokens.stops puts the model's own end-of-turn FIRST
        // and RequestPolicy.endTurn emits iterator().next(). Set.copyOf would salt-randomize
        // that pick per JVM run - a grammar dead end could then emit a turn HEADER
        Set<Integer> ordered = new LinkedHashSet<>(List.of(7, 3, 11, 5));
        LoadedModel<?> m =
                new LoadedModel<>(
                        new StubModel(),
                        StubTokenizer.encodingEvery(1),
                        "{{ messages }}",
                        ordered,
                        new byte[] {1, 2, 3, 4},
                        Optional.empty(),
                        LoadedModel.SamplingDefaults.NONE);
        assertIterableEquals(
                List.of(7, 3, 11, 5), m.stopTokens(), "insertion order is the contract");
        assertThrows(UnsupportedOperationException.class, () -> m.stopTokens().add(9));
    }

    private static LoadedModel<?> loaded(Tokenizer tokenizer) {
        return new LoadedModel<>(
                new StubModel(),
                tokenizer,
                "{{ messages }}",
                Set.of(2),
                new byte[] {1, 2, 3, 4},
                Optional.empty(),
                LoadedModel.SamplingDefaults.NONE);
    }

    /** Never runs: the record only needs a non-null model with a stable identity. */
    private static final class StubModel implements LanguageModel<Config, Void, RuntimeState> {
        @Override
        public Config config() {
            throw new UnsupportedOperationException();
        }

        @Override
        public Void weights() {
            throw new UnsupportedOperationException();
        }

        @Override
        public RuntimeState newState(int contextCapacity, int batchCapacity, Arena arena) {
            throw new UnsupportedOperationException();
        }

        @Override
        public void forward(RuntimeState state, Batch batch) {
            throw new UnsupportedOperationException();
        }

        @Override
        public FloatTensor head(RuntimeState state, int output) {
            throw new UnsupportedOperationException();
        }
    }

    /** Encodes each character to {@code stride * codePoint}: a knob that changes the ids only. */
    private record StubTokenizer(int stride) implements Tokenizer {

        static Tokenizer encodingEvery(int stride) {
            return new StubTokenizer(stride);
        }

        @Override
        public Vocabulary vocabulary() {
            return new Vocabulary() {
                @Override
                public int size() {
                    return 1000;
                }

                @Override
                public String token(int id) {
                    return String.valueOf(id);
                }

                @Override
                public int id(String token) {
                    return Integer.parseInt(token);
                }

                @Override
                public boolean contains(int id) {
                    return true;
                }

                @Override
                public boolean contains(String token) {
                    return true;
                }

                @Override
                public Iterator<Map.Entry<String, Integer>> iterator() {
                    return Collections.emptyIterator();
                }
            };
        }

        @Override
        public void encodeInto(CharSequence text, int from, int to, IntSequence.Builder out) {
            for (int i = from; i < to; i++) out.add(stride * text.charAt(i));
        }

        @Override
        public int decodeBytesInto(IntSequence ids, int from, ByteBuffer out) {
            throw new UnsupportedOperationException();
        }

        @Override
        public int countTokens(CharSequence text, int from, int to) {
            return to - from;
        }
    }
}
