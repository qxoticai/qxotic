package com.qxotic.jinfer.chat;

import static com.qxotic.jinfer.chat.ReplyLanguage.bytes;
import static com.qxotic.jinfer.chat.ReplyLanguage.call;
import static com.qxotic.jinfer.chat.ReplyLanguage.gbnf;
import static com.qxotic.jinfer.chat.ReplyLanguage.mark;
import static com.qxotic.jinfer.chat.ReplyLanguage.opt;
import static com.qxotic.jinfer.chat.ReplyLanguage.seq;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Config;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.LanguageModel;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * The {@code RequestPolicy.forceCall} dispatch, unit-level: a template declaring a {@code
 * forcedCallLanguage} is driven by ONE walk (seed = the selection's forced prefix, sampler =
 * mask-then-feed, parser seed = reply seed + forced prefix), while a template with only the legacy
 * hooks keeps the seed/pin recipe. The generation below runs the sampler exactly as the engine
 * would and must produce the family's own wire deterministically - forced regions are single-path
 * under the mask, the payload is the grammar's, and the reply ends with the model's own stop.
 */
public final class RequestPolicyForceCallTest {

    // ids 0..2 specials (the stop <end> FIRST: it is the endTurn the stop-set order promises)
    static final String[] SPECIALS = {"<end>", "<call>", "</call>"};
    static final String CHARS = "f(1)x";
    static final int END = 0, CALL = 1, END_CALL = 2;

    static int ch(char c) {
        return SPECIALS.length + CHARS.indexOf(c);
    }

    static final Tokenizer TOK = new FakeTokenizer();

    static final Tool F = new Tool("f", "{\"name\":\"f\",\"parameters\":{\"type\":\"object\"}}");

    /** A template whose forcing is the reply language; encode/parser are not exercised here. */
    static ChatTemplate languageTemplate() {
        return new ChatTemplate() {
            @Override
            public List<Batch> encode(Conversation conversation) {
                throw new UnsupportedOperationException();
            }

            @Override
            public ReplyParser parser() {
                throw new UnsupportedOperationException();
            }

            @Override
            public Optional<ReplyLanguage.Node> forcedCallLanguage(List<Tool> tools) {
                return Optional.of(
                        seq(
                                call(
                                        text -> List.of(new Part.ToolCall("", "f", Map.of())),
                                        mark("<call>"),
                                        bytes("f("),
                                        gbnf("root ::= \"1\""),
                                        bytes(")"),
                                        mark("</call>")),
                                opt(mark("<end>"))));
            }
        };
    }

    static LoadedModel<?> model(ChatTemplate template) {
        Set<Integer> stops = new LinkedHashSet<>(List.of(END));
        return new LoadedModel<>(
                new StubModel(),
                TOK,
                "{{ messages }}",
                stops,
                new byte[] {1},
                Optional.of(template),
                LoadedModel.SamplingDefaults.NONE);
    }

    @Test
    void theLanguagePathSeedsMasksAndEndsWithTheModelsOwnStop() {
        Sampler base = FloatTensor::argmax; // uniform logits: picks the first admissible token
        int[] replySeed = {ch('x')}; // a pretend prompt-opened tail, must survive in parserSeed
        RequestPolicy.ForcedCall forced =
                RequestPolicy.forceCall(model(languageTemplate()), List.of(F), base, replySeed)
                        .orElseThrow();

        // the seed is the selection's forced prefix: marker + the envelope bytes before the
        // payload, canonically tokenized - injected as prefill, not sampled
        int[] seed = ((Batch.Input.Tokens) forced.seed().input()).ids();
        assertArrayEquals(new int[] {CALL, ch('f'), ch('(')}, seed);

        // parser seed = reply seed ++ forced seed, the state the prompt leaves the parser in
        assertArrayEquals(new int[] {ch('x'), CALL, ch('f'), ch('(')}, forced.parserSeed());

        // drive the sampler as the engine would: every step masks, samples, feeds the walk.
        // Single-path regions force through the mask; the payload is the grammar's; after the
        // call the reply accepts and the model's own stop is the first admissible special;
        // anything past that is the dead-end contract: endTurn, the stop set's FIRST id.
        List<Integer> sampled = new ArrayList<>();
        F32FloatTensor logits = F32FloatTensor.allocate(Arena.ofAuto(), TOK.vocabulary().size());
        for (int i = 0; i < 6; i++) {
            for (int t = 0; t < TOK.vocabulary().size(); t++) logits.setFloat(t, 0f);
            sampled.add(forced.sampler().sampleToken(logits));
        }
        assertEquals(
                List.of(ch('1'), ch(')'), END_CALL, END, END, END),
                sampled,
                "payload, close, the terminator, then the dead-end endTurn forever");
    }

    @Test
    void aTemplateWithoutALanguageKeepsTheLegacySeedRecipe() {
        ChatTemplate legacy =
                new ChatTemplate() {
                    @Override
                    public List<Batch> encode(Conversation conversation) {
                        throw new UnsupportedOperationException();
                    }

                    @Override
                    public ReplyParser parser() {
                        throw new UnsupportedOperationException();
                    }

                    @Override
                    public int[] callSeed() {
                        return new int[] {CALL};
                    }
                };
        RequestPolicy.ForcedCall forced =
                RequestPolicy.forceCall(model(legacy), List.of(F), FloatTensor::argmax, new int[0])
                        .orElseThrow();
        assertArrayEquals(
                new int[] {CALL},
                ((Batch.Input.Tokens) forced.seed().input()).ids(),
                "the legacy path seeds the call marker alone");
    }

    @Test
    void aTemplateWithNeitherCannotForce() {
        ChatTemplate none =
                new ChatTemplate() {
                    @Override
                    public List<Batch> encode(Conversation conversation) {
                        throw new UnsupportedOperationException();
                    }

                    @Override
                    public ReplyParser parser() {
                        throw new UnsupportedOperationException();
                    }
                };
        assertTrue(
                RequestPolicy.forceCall(model(none), List.of(F), FloatTensor::argmax, new int[0])
                        .isEmpty());
    }

    // ---- fixtures ----------------------------------------------------------

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

    private static final class FakeTokenizer implements Tokenizer {
        private final Vocabulary vocab = new FakeVocabulary();

        @Override
        public Vocabulary vocabulary() {
            return vocab;
        }

        @Override
        public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
            for (int i = start; i < end; i++) out.add(ch(text.charAt(i)));
        }

        @Override
        public int countTokens(CharSequence text, int start, int end) {
            return end - start;
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
            if (tokenStartIndex == tokens.length()) return 0;
            int id = tokens.intAt(tokenStartIndex);
            String piece =
                    id < SPECIALS.length
                            ? SPECIALS[id]
                            : String.valueOf(CHARS.charAt(id - SPECIALS.length));
            out.put(piece.getBytes(java.nio.charset.StandardCharsets.UTF_8));
            return 1;
        }
    }

    private static final class FakeVocabulary implements Vocabulary {
        @Override
        public int size() {
            return SPECIALS.length + CHARS.length();
        }

        @Override
        public String token(int id) {
            return id < SPECIALS.length
                    ? SPECIALS[id]
                    : String.valueOf(CHARS.charAt(id - SPECIALS.length));
        }

        @Override
        public int id(String text) {
            for (int i = 0; i < SPECIALS.length; i++) {
                if (SPECIALS[i].equals(text)) return i;
            }
            int at = text.length() == 1 ? CHARS.indexOf(text.charAt(0)) : -1;
            if (at < 0) throw new NoSuchElementException(text);
            return SPECIALS.length + at;
        }

        @Override
        public boolean contains(int id) {
            return id >= 0 && id < size();
        }

        @Override
        public boolean contains(String text) {
            try {
                id(text);
                return true;
            } catch (NoSuchElementException e) {
                return false;
            }
        }

        @Override
        public boolean isTokenOfType(int id, TokenType type) {
            if (!contains(id)) throw new NoSuchElementException("id " + id);
            boolean special = id < SPECIALS.length;
            if (type == StandardTokenType.NORMAL) return !special;
            if (type == StandardTokenType.CONTROL) return special;
            return false;
        }

        @Override
        public Iterator<Map.Entry<String, Integer>> iterator() {
            return Collections.emptyIterator();
        }
    }
}
