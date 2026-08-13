package com.qxotic.jinfer.x.chat;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.Media;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Test;

final class ChatApiTest {
    private static final ByteTokenizer TOKENIZER = new ByteTokenizer();

    @Test
    void promptWriterPreservesRunsAndBoundsBatches() {
        TOKENIZER.takeEncoded();
        List<Batch> batches = new ArrayList<>();
        PromptWriter writer = new PromptWriter(TOKENIZER, 2, batches::add);
        writer.text("a").trusted("b<eot>c").text("d").finish();

        assertEquals(List.of("ab", "cd"), TOKENIZER.takeEncoded());
        assertArrayEquals(TOKENIZER.ids("ab", ByteTokenizer.EOT, "cd"), Batch.tokenIds(batches));
        assertTrue(batches.stream().allMatch(batch -> batch.count() <= 2));
    }

    @Test
    void trustedTextRecognizesPrefixingSpecialSpellings() {
        List<Batch> batches = new ArrayList<>();
        new PromptWriter(TOKENIZER, 32, batches::add).trusted("<x<x-long>").finish();

        assertArrayEquals(
                new int[] {ByteTokenizer.X, ByteTokenizer.X_LONG}, Batch.tokenIds(batches));
    }

    @Test
    void mediaRowsAreStreamedWithTheirKey() {
        List<Batch> batches = new ArrayList<>();
        byte[] key = {1, 2, 3};
        Media.Image image = new Media.Image(new float[] {0, 0, 0}, 1, 1, 3);
        try (Arena arena = Arena.ofConfined()) {
            PromptWriter writer = new PromptWriter(TOKENIZER, 2, batches::add);
            writer.text("x");
            writer.media(
                    image,
                    key,
                    (source, max, sink) ->
                            sink.accept(Views.allocateF32(new PanamaMemoryArena(arena), 2, 4)),
                    true);
            writer.text("y").finish();
        }

        assertEquals(3, batches.size());
        Batch.Input.Embeddings media =
                assertInstanceOf(Batch.Input.Embeddings.class, batches.get(1).input());
        assertArrayEquals(key, media.contentKey());
        assertTrue(media.bidirectional());
    }

    @Test
    void seedSetsReasoningStateWithoutLeakingPromptText() {
        ReplyParser parser = ReplyParser.spans(TOKENIZER);
        parser.seed(IntSequence.of(ByteTokenizer.THINK_OPEN).concat(TOKENIZER.encode("prompt")));

        assertEquals(Channel.REASONING, parser.channel());
        TOKENIZER.encode("thought").forEachInt(parser::feed);
        parser.feed(ByteTokenizer.THINK_CLOSE);
        TOKENIZER.encode("answer").forEachInt(parser::feed);

        Message reply = parser.finish();
        Content.Reasoning reasoning =
                assertInstanceOf(Content.Reasoning.class, reply.content().get(0));
        assertEquals("thought", reasoning.text());
        assertEquals("answer", assertInstanceOf(Content.Text.class, reply.content().get(1)).text());
    }

    @Test
    void forcedCallSeedKeepsOpenCapture() {
        ReplyParser parser =
                ReplyParser.spans(
                        TOKENIZER,
                        "<call>",
                        "</call>",
                        text -> List.of(new Content.ToolCall("", text, Map.of())));
        parser.seed(IntSequence.of(ByteTokenizer.CALL_OPEN).concat(TOKENIZER.encode("wea")));
        TOKENIZER.encode("ther").forEachInt(parser::feed);
        parser.feed(ByteTokenizer.CALL_CLOSE);

        Content.ToolCall call =
                assertInstanceOf(Content.ToolCall.class, parser.finish().content().getFirst());
        assertEquals("weather", call.name());
        assertArrayEquals(TOKENIZER.encode("weather").toArray(), call.verbatim().toArray());
    }

    @Test
    void droppingAlsoFiltersPromptOwnedTokens() {
        ReplyParser parser =
                ReplyParser.dropping(ReplyParser.spans(TOKENIZER), ByteTokenizer.THINK_CLOSE);
        parser.seed(IntSequence.of(ByteTokenizer.THINK_OPEN, ByteTokenizer.THINK_CLOSE));

        assertEquals(Channel.REASONING, parser.channel());
    }

    @Test
    void droppingPreservesParserLifecycle() {
        ReplyParser parser =
                ReplyParser.dropping(ReplyParser.spans(TOKENIZER), ByteTokenizer.THINK_CLOSE);
        assertEquals("", parser.feed(ByteTokenizer.THINK_CLOSE).text());
        assertThrows(IllegalStateException.class, () -> parser.seed(IntSequence.empty()));
        parser.finish();
        assertThrows(IllegalStateException.class, () -> parser.feed(ByteTokenizer.THINK_CLOSE));
    }

    private static final class ByteTokenizer implements Tokenizer {
        static final int BOS = 0;
        static final int START_HEADER = 1;
        static final int END_HEADER = 2;
        static final int EOT = 3;
        static final int THINK_OPEN = 4;
        static final int THINK_CLOSE = 5;
        static final int CALL_OPEN = 6;
        static final int CALL_CLOSE = 7;
        static final int X = 8;
        static final int X_LONG = 9;
        private static final int BYTE_OFFSET = 10;
        private static final List<String> SPECIALS =
                List.of(
                        "<bos>",
                        "<header>",
                        "</header>",
                        "<eot>",
                        "<think>",
                        "</think>",
                        "<call>",
                        "</call>",
                        "<x",
                        "<x-long>");

        private final Vocabulary vocabulary = new ByteVocabulary();
        private final List<String> encoded = new ArrayList<>();

        @Override
        public Vocabulary vocabulary() {
            return vocabulary;
        }

        @Override
        public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
            String value = text.subSequence(start, end).toString();
            encoded.add(value);
            for (byte next : value.getBytes(StandardCharsets.UTF_8))
                out.add(BYTE_OFFSET + (next & 255));
        }

        @Override
        public int countTokens(CharSequence text, int start, int end) {
            return text.subSequence(start, end).toString().getBytes(StandardCharsets.UTF_8).length;
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int from, ByteBuffer out) {
            if (from == tokens.length()) return 0;
            int token = tokens.intAt(from);
            byte[] bytes =
                    token < BYTE_OFFSET
                            ? SPECIALS.get(token).getBytes(StandardCharsets.UTF_8)
                            : new byte[] {(byte) (token - BYTE_OFFSET)};
            out.put(bytes);
            return 1;
        }

        List<String> takeEncoded() {
            List<String> copy = List.copyOf(encoded);
            encoded.clear();
            return copy;
        }

        int[] ids(Object... parts) {
            IntSequence.Builder out = IntSequence.newBuilder();
            for (Object part : parts) {
                if (part instanceof Integer token) out.add(token);
                else
                    for (byte next : ((String) part).getBytes(StandardCharsets.UTF_8))
                        out.add(BYTE_OFFSET + (next & 255));
            }
            return out.build().toArray();
        }

        private static final class ByteVocabulary implements Vocabulary {
            @Override
            public int size() {
                return BYTE_OFFSET + 256;
            }

            @Override
            public String token(int id) {
                if (!contains(id)) throw new NoSuchElementException("id " + id);
                return id < BYTE_OFFSET ? SPECIALS.get(id) : "byte:" + (id - BYTE_OFFSET);
            }

            @Override
            public int id(String text) {
                int id = SPECIALS.indexOf(text);
                if (id < 0) throw new NoSuchElementException(text);
                return id;
            }

            @Override
            public boolean contains(int id) {
                return id >= 0 && id < size();
            }

            @Override
            public boolean contains(String text) {
                return SPECIALS.contains(text);
            }

            @Override
            public boolean isTokenOfType(int id, TokenType type) {
                if (!contains(id)) throw new NoSuchElementException("id " + id);
                if (type == StandardTokenType.NORMAL) return id >= BYTE_OFFSET;
                if (type == StandardTokenType.CONTROL) return id < BYTE_OFFSET;
                return false;
            }

            @Override
            public Iterator<Map.Entry<String, Integer>> iterator() {
                return IntStream.range(0, size())
                        .<Map.Entry<String, Integer>>mapToObj(i -> Map.entry(token(i), i))
                        .iterator();
            }
        }
    }
}
