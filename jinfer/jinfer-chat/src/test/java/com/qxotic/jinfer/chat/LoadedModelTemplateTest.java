package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Config;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.toknroll.Tokenizer;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * The template-override contract of {@link LoadedModel#withChatTemplateSource} and {@link
 * LoadedModel#withTemplate}: a source override drops the native codec (it was ported from the
 * container's template - framing a custom wire with it would encode conversations the model never
 * saw), a {@link ChatTemplate} override takes the native slot, and NEITHER re-roots the cache seed
 * (a different template renders different token streams, so stale cached prefixes miss cold -
 * unlike a tokenizer swap, which changes what identical ids MEAN).
 */
final class LoadedModelTemplateTest {

    @Test
    void sourceOverrideReplacesTheSourceAndDropsTheNativeCodec() {
        ChatTemplate nativePort = new StubTemplate();
        LoadedModel<?> base = loaded(Optional.of(nativePort));
        LoadedModel<?> derived = base.withChatTemplateSource("{{ my_custom_wire }}");

        assertEquals("{{ my_custom_wire }}", derived.chatTemplateSource());
        assertTrue(
                derived.template().isEmpty(),
                "the native codec was ported from the CONTAINER's template; a custom wire must"
                        + " not be framed with it");
        assertEquals(Optional.of(nativePort), base.template(), "the original record is untouched");
        assertArrayEquals(base.seed(), derived.seed(), "a template change never re-roots the seed");
        assertSame(base.model(), derived.model());
        assertSame(base.tokenizer(), derived.tokenizer());
    }

    @Test
    void templateOverrideTakesTheNativeSlot() {
        LoadedModel<?> base = loaded(Optional.empty());
        ChatTemplate custom = new StubTemplate();
        LoadedModel<?> derived = base.withTemplate(custom);

        assertEquals(Optional.of(custom), derived.template());
        assertSame(
                base.chatTemplateSource(),
                derived.chatTemplateSource(),
                "the container's Jinja stays as the punt fallback");
        assertArrayEquals(base.seed(), derived.seed(), "a template change never re-roots the seed");
    }

    @Test
    void invalidOverridesAreRejected() {
        LoadedModel<?> base = loaded(Optional.empty());
        assertThrows(IllegalArgumentException.class, () -> base.withChatTemplateSource(null));
        assertThrows(IllegalArgumentException.class, () -> base.withChatTemplateSource("  "));
        assertThrows(IllegalArgumentException.class, () -> base.withTemplate(null));
    }

    private static LoadedModel<?> loaded(Optional<ChatTemplate> template) {
        return new LoadedModel<>(
                new StubModel(),
                StubTokenizer.INSTANCE,
                "{{ container_wire }}",
                Set.of(2),
                new byte[] {1, 2, 3, 4},
                template,
                SamplingDefaults.NONE);
    }

    /** Never encodes: the record only needs a stable identity in the template slot. */
    private static final class StubTemplate implements ChatTemplate {
        @Override
        public List<Batch> encode(Conversation conversation) {
            throw new UnsupportedOperationException();
        }

        @Override
        public ReplyParser parser() {
            throw new UnsupportedOperationException();
        }

        @Override
        public int[] replySeed(boolean thinking) {
            throw new UnsupportedOperationException();
        }
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
        public RuntimeState newState(
                int contextCapacity, int batchCapacity, java.lang.foreign.Arena arena) {
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

    /** Never tokenizes: the record only needs a non-null tokenizer. */
    private enum StubTokenizer implements Tokenizer {
        INSTANCE;

        @Override
        public com.qxotic.toknroll.Vocabulary vocabulary() {
            throw new UnsupportedOperationException();
        }

        @Override
        public void encodeInto(
                CharSequence text, int from, int to, com.qxotic.toknroll.IntSequence.Builder out) {
            throw new UnsupportedOperationException();
        }

        @Override
        public int decodeBytesInto(
                com.qxotic.toknroll.IntSequence ids, int from, java.nio.ByteBuffer out) {
            throw new UnsupportedOperationException();
        }

        @Override
        public int countTokens(CharSequence text, int from, int to) {
            throw new UnsupportedOperationException();
        }
    }
}
