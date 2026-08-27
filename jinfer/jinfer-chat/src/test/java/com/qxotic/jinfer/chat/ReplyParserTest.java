package com.qxotic.jinfer.chat;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;
import java.util.concurrent.atomic.AtomicReference;

/**
 * The span {@link ReplyParser} stepper over a fake vocabulary: channel routing on trusted think
 * ids, atomic call spans (nothing leaks mid-span; calls only in finish()), scaffold inertness,
 * split-UTF-8 buffering, and the streamed/one-shot agreement law.
 */
public final class ReplyParserTest {

    // ids:            0          1           2         3          4        5       6      7
    static final String[] W = {
        "<think>",
        "</think>",
        "<|call|>",
        "<|/call|>",
        "<|end|>",
        "Hello",
        " world",
        "[f(x=1)]",
        "é-h1",
        "é-h2",
        "bad",
        "[f(",
        "x",
        " ",
        "=1)]",
        "{\"name\": \"f\",",
        " \"arguments\": {\"x\": 1",
        ",},}",
        "\n\n",
        "Hi"
    };
    // 18: the blank line every reasoning family puts after </think>; 19: a word after it
    // 8, 9: the two halves of a split two-byte UTF-8 code point (é = 0xC3 0xA9); 10: malformed;
    // 11-14: one call payload fragmented four ways, incl. a whitespace-only fragment;
    // 15-17: a JSON envelope with trailing commas (the salvage lane's payload, fragmented)
    static final byte[][] BYTES = {
        b("<think>"),
        b("</think>"),
        b("<|call|>"),
        b("<|/call|>"),
        b("<|end|>"),
        b("Hello"),
        b(" world"),
        b("[f(x=1)]"),
        new byte[] {(byte) 0xC3},
        new byte[] {(byte) 0xA9},
        new byte[] {(byte) 0xFF},
        b("[f("),
        b("x"),
        b(" "),
        b("=1)]"),
        b("{\"name\": \"f\","),
        b(" \"arguments\": {\"x\": 1"),
        b(",},}"),
        b("\n\n"),
        b("Hi")
    };
    static final int SPECIALS = 5; // ids 0..4 are special

    static final Tokenizer TOK = new FakeTokenizer();

    static void check(boolean ok, String what) {
        Assertions.assertTrue(ok, what);
    }

    static byte[] b(String s) {
        return s.getBytes(StandardCharsets.UTF_8);
    }

    record Step(String fragment, boolean reasoning) {}

    static List<Step> run(ReplyParser p, int... tokens) {
        List<Step> steps = new ArrayList<>();
        for (int t : tokens) {
            ReplyParser.Fragment f = p.feed(t);
            if (!f.text().isEmpty()) steps.add(new Step(f.text(), p.reasoning()));
        }
        return steps;
    }

    static List<Content.ToolCall> parseCalls(String payload) {
        return "[f(x=1)]".equals(payload)
                ? List.of(new Content.ToolCall("", "f", Map.of("x", 1)))
                : List.of();
    }

    @Test
    void whitespaceAfterAThinkCloseIsFramingNotContent() {
        // "</think>\n\nHi": the blank line frames the answer (llama.cpp's parser consumes it
        // too); the streamed text and the finished message both start at "Hi", the ids stay
        ReplyParser p = ReplyParser.spans(TOK);
        List<Step> steps = run(p, 0, 5, 1, 18, 19);
        Assertions.assertEquals(List.of(new Step("Hello", true), new Step("Hi", false)), steps);
        Message m = p.finish();
        Assertions.assertEquals("Hi", ((Content.Text) m.content().get(1)).text());
        Assertions.assertEquals(
                2, ((Content.Text) m.content().get(1)).verbatim().length(), "ids verbatim");
    }

    @Test
    void spanGrammarStepper() {
        // 1. plain content: fragments stream on the content channel, coalesce in the message
        ReplyParser p = ReplyParser.spans(TOK);
        List<Step> steps = run(p, 5, 6, 4); // Hello, " world", <|end|> (scaffold)
        check(
                steps.equals(List.of(new Step("Hello", false), new Step(" world", false))),
                "content fragments stream; scaffold is inert: " + steps);
        Message m = p.finish();
        check(m.text().equals("Hello world"), "message coalesces text: " + m.text());
        check(
                m.content().size() == 1
                        && ((Content.Text) m.content().get(0))
                                .verbatim()
                                .toList()
                                .equals(List.of(5, 6)),
                "coalesced text carries verbatim payload ids");

        // 2. think span: markers hidden, channel flips, unterminated span closes at finish
        p = ReplyParser.spans(TOK);
        steps = run(p, 0, 5, 1, 6); // <think>Hello</think> world
        // the space after </think> is the answer's frame, not its first character
        check(
                steps.equals(List.of(new Step("Hello", true), new Step("world", false))),
                "think routing + channel flags: " + steps);
        m = p.finish();
        check(
                m.content().get(0) instanceof Content.Reasoning r
                        && r.content().size() == 1
                        && "Hello".equals(((Content.Text) r.content().get(0)).text()),
                "reasoning tree in message");
        p = ReplyParser.spans(TOK);
        run(p, 0, 5); // unterminated think
        m = p.finish();
        check(
                m.content().size() == 1 && m.content().get(0) instanceof Content.Reasoning,
                "unterminated think span is still reasoning");

        // 3. call span: atomic - nothing streams, call surfaces in finish() with verbatim ids
        p = ReplyParser.spans(TOK, "<|call|>", "<|/call|>", ReplyParserTest::parseCalls);
        steps = run(p, 5, 2, 7, 3, 4); // Hello <|call|>[f(x=1)]<|/call|> <|end|>
        check(
                steps.equals(List.of(new Step("Hello", false))),
                "call span never reaches the text stream: " + steps);
        m = p.finish();
        List<Content.ToolCall> calls = new ArrayList<>();
        for (Content part : m.content()) if (part instanceof Content.ToolCall c) calls.add(c);
        check(
                calls.size() == 1
                        && "f".equals(calls.get(0).name())
                        && calls.get(0).arguments().get("x") instanceof Number n
                        && n.intValue() == 1,
                "call parsed structurally in finish(): " + calls);
        check(
                calls.get(0).verbatim() != null
                        && calls.get(0).verbatim().toList().equals(List.of(7)),
                "call carries its payload verbatim ids");

        // 3b. the same span stays visible when calls are not claimed, while its hidden markers
        // remain in verbatim so extending the conversation can replay the exact generated wire
        p =
                ReplyParser.spans(
                        TOK,
                        "<|call|>",
                        "<|/call|>",
                        ReplyParserTest::parseCalls,
                        "<think>",
                        "</think>",
                        false);
        steps = run(p, 2, 7, 3);
        check(
                steps.equals(List.of(new Step("[f(x=1)]", false))),
                "unclaimed call payload is visible: " + steps);
        m = p.finish();
        Content.Text unclaimed = (Content.Text) m.content().getFirst();
        check(unclaimed.text().equals("[f(x=1)]"), "unclaimed span becomes text");
        check(
                unclaimed.verbatim().toList().equals(List.of(2, 7, 3)),
                "unclaimed text retains the complete call wire");

        // 3c. a claimed but malformed span remains visible rather than disappearing
        p = ReplyParser.spans(TOK, "<|call|>", "<|/call|>", ReplyParserTest::parseCalls);
        steps = run(p, 2, 5, 3);
        check(
                steps.equals(List.of(new Step("Hello", false))),
                "malformed call payload is visible: " + steps);
        m = p.finish();
        Content.Text malformed = (Content.Text) m.content().getFirst();
        check(malformed.verbatim().toList().equals(List.of(2, 5, 3)), "malformed wire replays");

        // 3d. a new opener discards an unterminated candidate instead of claiming it
        p = ReplyParser.spans(TOK, "<|call|>", "<|/call|>", ReplyParserTest::parseCalls);
        run(p, 2, 7, 2, 7, 3);
        m = p.finish();
        check(
                m.content().stream().filter(Content.ToolCall.class::isInstance).count() == 1,
                "only the closed call is claimed");

        // 4. a span the generation never closed is NO call
        p = ReplyParser.spans(TOK, "<|call|>", "<|/call|>", ReplyParserTest::parseCalls);
        run(p, 2, 7); // <|call|>[f(x=1)] ... never closed
        m = p.finish();
        check(
                m.content().stream().noneMatch(part -> part instanceof Content.ToolCall),
                "unclosed call span parses to no call");

        // 5. split UTF-8: no fragment until the code point completes, then it arrives whole
        p = ReplyParser.spans(TOK);
        check(p.feed(8).text().isEmpty(), "first half of a split code point buffers");
        check("é".equals(p.feed(9).text()), "second half completes the code point");

        // 6. malformed bytes are replaced immediately rather than accumulating forever
        p = ReplyParser.spans(TOK);
        check("�".equals(p.feed(10).text()), "malformed UTF-8 emits a replacement");
        check(
                "Hello".equals(p.feed(5).text()),
                "valid text after malformed UTF-8 remains independent");

        // 7. streamed and one-shot agree
        IntSequence reply = IntSequence.of(0, 5, 1, 6, 4);
        Message streamed = ReplyParser.parse(ReplyParser.spans(TOK), reply);
        Message oneshot = ReplyParser.parse(ReplyParser.spans(TOK), reply);
        check(streamed.equals(oneshot), "streamed == one-shot decode");
    }

    @Test
    void sloppyJsonCallIsSalvagedThroughTheSpanParser() {
        // the wiring pin, one level above ToolCallSyntaxTest: the JSON-envelope families (SmolLM3,
        // Granite) hand ToolCallSyntax.parseBlock to the span parser with NO schema grammar on the
        // arguments - a trailing-comma payload must still surface as a call, fragmented or not,
        // where a strict parse would drop the span to visible text and the tool loop would die
        ReplyParser p = ReplyParser.spans(TOK, "<|call|>", "<|/call|>", ToolCallSyntax::parseBlock);
        List<Step> steps = run(p, 5, 2, 15, 16, 17, 3, 4);
        check(
                steps.equals(List.of(new Step("Hello", false))),
                "the call span never streams: " + steps);
        Message m = p.finish();
        List<Content.ToolCall> calls = new ArrayList<>();
        for (Content part : m.content()) if (part instanceof Content.ToolCall c) calls.add(c);
        check(calls.size() == 1, "one call salvaged, not dropped: " + m.content());
        check("f".equals(calls.get(0).name()), "the envelope name survives: " + calls.get(0));
        check(
                calls.get(0).arguments().get("x") instanceof Number n && n.intValue() == 1,
                "trailing commas salvaged to {x: 1}: " + calls.get(0).arguments());
        check(
                calls.get(0).verbatim().toList().equals(List.of(15, 16, 17)),
                "the salvaged call keeps its sloppy payload verbatim for exact replay");
    }

    @Test
    void fragmentedCallSpanDeliversThePayloadVerbatim() {
        // a call payload fragmented across many tokens - a whitespace-only fragment included -
        // must reach the call parser as the EXACT concatenation, with nothing streamed mid-span
        // (langchain4j's blank-partial-arguments case, at jinfer's boundary: spans are atomic)
        AtomicReference<String> received =
                new AtomicReference<>();
        ReplyParser p =
                ReplyParser.spans(
                        TOK,
                        "<|call|>",
                        "<|/call|>",
                        payload -> {
                            received.set(payload);
                            return List.of(new Content.ToolCall("", "f", Map.of("x", 1)));
                        });
        List<Step> steps = run(p, 2, 11, 12, 13, 14, 3, 4);
        check(steps.isEmpty(), "nothing streams from inside a call span: " + steps);
        check(
                "[f(x =1)]".equals(received.get()),
                "payload is the exact fragment concatenation, whitespace kept: '"
                        + received.get()
                        + "'");
        Message m = p.finish();
        check(
                m.content().size() == 1 && m.content().get(0) instanceof Content.ToolCall,
                "exactly one call claimed: " + m.content());
        Content.ToolCall call = (Content.ToolCall) m.content().get(0);
        check(
                call.verbatim() != null && call.verbatim().toList().equals(List.of(11, 12, 13, 14)),
                "the call carries every fragment's verbatim ids");
    }

    private static final class FakeTokenizer implements Tokenizer {
        private final Vocabulary vocab = new FakeVocabulary();

        @Override
        public Vocabulary vocabulary() {
            return vocab;
        }

        @Override
        public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
            throw new UnsupportedOperationException("decode-only fake");
        }

        @Override
        public int countTokens(CharSequence text, int start, int end) {
            throw new UnsupportedOperationException("decode-only fake");
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
            if (tokenStartIndex == tokens.length()) return 0;
            out.put(BYTES[tokens.intAt(tokenStartIndex)]);
            return 1;
        }
    }

    private static final class FakeVocabulary implements Vocabulary {
        @Override
        public int size() {
            return W.length + 3;
        }

        @Override
        public String token(int id) {
            if (id < W.length) return W[id];
            if (contains(id)) return "<byte>";
            throw new NoSuchElementException("id " + id);
        }

        @Override
        public int id(String text) {
            for (int i = 0; i < W.length; i++) if (W[i].equals(text)) return i;
            throw new NoSuchElementException(text);
        }

        @Override
        public boolean contains(int id) {
            return id >= 0 && id < size();
        }

        @Override
        public boolean contains(String text) {
            for (String w : W) if (w.equals(text)) return true;
            return false;
        }

        @Override
        public boolean isTokenOfType(int id, TokenType type) {
            if (!contains(id)) throw new NoSuchElementException("id " + id);
            boolean special = id < SPECIALS;
            if (type == StandardTokenType.NORMAL) return !special;
            if (type == StandardTokenType.CONTROL) return special;
            return false;
        }

        @Override
        public Iterator<Map.Entry<String, Integer>> iterator() {
            List<Map.Entry<String, Integer>> entries = new ArrayList<>();
            for (int i = 0; i < W.length; i++) entries.add(Map.entry(W[i], i));
            return entries.iterator();
        }
    }

    @Test
    void aThinkCloseInsideAnOpenCallSpanClosesTheThought() {
        // the thinking cap forces </think> wherever the model is; inside a call span the
        // detector must not swallow it: the partial span stays reasoning text, the answer
        // after it is content
        ReplyParser p =
                ReplyParser.spans(TOK, "<|call|>", "<|/call|>", ReplyParserTest::parseCalls);
        List<Step> steps = run(p, 0, 5, 2, 11, 1, 19);
        Assertions.assertEquals(
                List.of(new Step("Hello", true), new Step("[f(", true), new Step("Hi", false)),
                steps);
        Message m = p.finish();
        Assertions.assertEquals("Hi", ((Content.Text) m.content().get(1)).text());
    }

    @Test
    void anUnterminatedUnclaimedSpanKeepsItsTextAndIds() {
        // no tools offered: a call span is visible text by contract, and a span cut by
        // maxTokens is still the model's text with the ids the cache already ingested
        ReplyParser p =
                ReplyParser.spans(
                        TOK,
                        "<|call|>",
                        "<|/call|>",
                        ReplyParserTest::parseCalls,
                        "<think>",
                        "</think>",
                        false);
        run(p, 5, 2, 11);
        Message m = p.finish();
        Content.Text text = (Content.Text) m.content().get(m.content().size() - 1);
        Assertions.assertEquals("[f(", text.text());
        Assertions.assertEquals(IntSequence.of(2, 11), text.verbatim(), "ids verbatim");
    }
}
