package com.qxotic.jinfer.chat;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

/**
 * The span {@link ReplyParser} stepper over a fake vocabulary: channel routing on trusted think
 * ids, atomic call spans (nothing leaks mid-span; calls only in finish()), scaffold inertness,
 * split-UTF-8 buffering, and the streamed/one-shot agreement law.
 */
public final class ReplyParserTest {

    // ids:            0          1           2         3          4        5       6      7
    static final String[] W = {
        "<think>", "</think>", "<|call|>", "<|/call|>", "<|end|>", "Hello", " world", "[f(x=1)]"
    };
    // 8, 9: the two halves of a split two-byte UTF-8 code point (é = 0xC3 0xA9)
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
        new byte[] {(byte) 0xA9}
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
            String s = p.feed(t);
            if (!s.isEmpty()) steps.add(new Step(s, p.reasoning()));
        }
        return steps;
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
                        && ((Part.Text) m.content().get(0))
                                .verbatim()
                                .toList()
                                .equals(List.of(5, 6)),
                "coalesced text carries verbatim payload ids");

        // 2. think span: markers hidden, channel flips, unterminated span closes at finish
        p = ReplyParser.spans(TOK);
        steps = run(p, 0, 5, 1, 6); // <think>Hello</think> world
        check(
                steps.equals(List.of(new Step("Hello", true), new Step(" world", false))),
                "think routing + channel flags: " + steps);
        m = p.finish();
        check(
                m.content().get(0) instanceof Part.Reasoning r
                        && r.content().size() == 1
                        && "Hello".equals(((Part.Text) r.content().get(0)).text()),
                "reasoning tree in message");
        p = ReplyParser.spans(TOK);
        run(p, 0, 5); // unterminated think
        m = p.finish();
        check(
                m.content().size() == 1 && m.content().get(0) instanceof Part.Reasoning,
                "unterminated think span is still reasoning");

        // 3. call span: atomic - nothing streams, call surfaces in finish() with verbatim ids
        p = ReplyParser.spans(TOK, "<|call|>", "<|/call|>", ToolCallSyntax::parseBlock);
        steps = run(p, 5, 2, 7, 3, 4); // Hello <|call|>[f(x=1)]<|/call|> <|end|>
        check(
                steps.equals(List.of(new Step("Hello", false))),
                "call span never reaches the text stream: " + steps);
        m = p.finish();
        List<Part.ToolCall> calls = new ArrayList<>();
        for (Part part : m.content()) if (part instanceof Part.ToolCall c) calls.add(c);
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

        // 4. a span the generation never closed is NO call
        p = ReplyParser.spans(TOK, "<|call|>", "<|/call|>", ToolCallSyntax::parseBlock);
        run(p, 2, 7); // <|call|>[f(x=1)] ... never closed
        m = p.finish();
        check(
                m.content().stream().noneMatch(part -> part instanceof Part.ToolCall),
                "unclosed call span parses to no call");

        // 5. split UTF-8: no fragment until the code point completes, then it arrives whole
        p = ReplyParser.spans(TOK);
        check(p.feed(8).isEmpty(), "first half of a split code point buffers");
        check("é".equals(p.feed(9)), "second half completes the code point");

        // 6. streamed and one-shot agree
        IntSequence reply = IntSequence.of(0, 5, 1, 6, 4);
        Message streamed = ReplyParser.parse(ReplyParser.spans(TOK), reply);
        Message oneshot = ReplyParser.parse(ReplyParser.spans(TOK), reply);
        check(streamed.equals(oneshot), "streamed == one-shot decode");
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
            return W.length + 2;
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
        public java.util.Iterator<Map.Entry<String, Integer>> iterator() {
            List<Map.Entry<String, Integer>> entries = new ArrayList<>();
            for (int i = 0; i < W.length; i++) entries.add(Map.entry(W[i], i));
            return entries.iterator();
        }
    }
}
