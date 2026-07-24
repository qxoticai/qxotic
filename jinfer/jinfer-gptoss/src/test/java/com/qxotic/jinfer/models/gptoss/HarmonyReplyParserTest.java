package com.qxotic.jinfer.models.gptoss;

import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyParser;
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
 * The Harmony channel grammar over a fake vocabulary: {@code
 * <|channel|>analysis<|message|>...<|end|><|start|>assistant<|channel|>final<|message|>...<|return|>}
 * routes analysis bodies to reasoning and final bodies to content, header text never leaks, and
 * finish() builds the structured message.
 */
public final class HarmonyReplyParserTest {

    // ids:           0            1             2          3           4
    static final String[] W = {
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|return|>",
        //   5            6
        "<|call|>",
        "<|constrain|>",
        //   7           8          9           10           11
        "assistant",
        "analysis",
        "final",
        "thinking...",
        "The answer is 4.",
        //   12           13                            14    15
        "commentary",
        " to=functions.get_weather",
        " ",
        "json",
        //   16              17        18
        "{\"city\": \"Pa",
        "ris\"}",
        "{broken"
    };
    static final int SPECIALS = 7; // ids 0..6 special; 7.. plain

    static final Tokenizer TOK = new FakeTokenizer();

    static void check(boolean ok, String what) {
        Assertions.assertTrue(ok, what);
    }

    record Step(String fragment, boolean reasoning) {}

    @Test
    void harmonyChannelRouting() {
        // <|channel|>analysis<|message|>thinking...<|end|><|start|>assistant<|channel|>final
        // <|message|>The answer is 4.<|return|>
        int[] reply = {1, 8, 2, 10, 3, 0, 7, 1, 9, 2, 11, 4};
        ReplyParser p = new HarmonyReplyParser(TOK);
        List<Step> steps = new ArrayList<>();
        for (int t : reply) {
            String s = p.feed(t);
            if (!s.isEmpty()) steps.add(new Step(s, p.reasoning()));
        }
        check(
                steps.equals(
                        List.of(
                                new Step("thinking...", true),
                                new Step("The answer is 4.", false))),
                "channel routing: analysis -> reasoning, final -> content: " + steps);

        Message m = p.finish();
        check(m.content().size() == 2, "message: reasoning node + text: " + m.content());
        check(
                m.content().get(0) instanceof Part.Reasoning r
                        && "thinking...".equals(((Part.Text) r.content().get(0)).text()),
                "reasoning tree holds the analysis body");
        check("The answer is 4.".equals(m.text()), "content text is the final body");
        check(
                m.content().get(1) instanceof Part.Text t
                        && t.verbatim().toList().equals(List.of(11)),
                "final body carries verbatim ids");

        // header text (role, channel names) never leaks into either channel
        String all = m.text() + m.content();
        check(!all.contains("assistant") && !all.contains("analysis"), "header text never leaks");

        // one-shot equals streamed
        Message oneshot = ReplyParser.parse(new HarmonyReplyParser(TOK), IntSequence.of(reply));
        check(oneshot.equals(m), "streamed == one-shot");
    }

    @Test
    void commentaryToolCall() {
        // <|channel|>analysis<|message|>thinking...<|end|><|start|>assistant<|channel|>commentary
        //  to=functions.get_weather <|constrain|>json<|message|>{"city": "Paris"}<|call|>
        int[] reply = {1, 8, 2, 10, 3, 0, 7, 1, 12, 13, 14, 6, 15, 2, 16, 17, 5};
        ReplyParser p = new HarmonyReplyParser(TOK);
        List<Step> steps = new ArrayList<>();
        for (int t : reply) {
            String s = p.feed(t);
            if (!s.isEmpty()) steps.add(new Step(s, p.reasoning()));
        }
        check(
                steps.equals(List.of(new Step("thinking...", true))),
                "call payload never streams: " + steps);

        Message m = p.finish();
        check(m.content().size() == 2, "message: reasoning + call: " + m.content());
        check(m.content().get(0) instanceof Part.Reasoning, "analysis body kept as reasoning");
        check(
                m.content().get(1) instanceof Part.ToolCall c
                        && "get_weather".equals(c.name())
                        && Map.of("city", "Paris").equals(c.arguments())
                        && c.verbatim().toList().equals(List.of(16, 17)),
                "call parsed with name, arguments, verbatim payload ids: " + m.content().get(1));
        check(m.text().isEmpty(), "call payload never reaches content");

        // stop token withheld by the driver: finish() still closes the open call body
        int[] unterminated = {1, 12, 13, 14, 6, 15, 2, 16, 17};
        Message open = ReplyParser.parse(new HarmonyReplyParser(TOK), IntSequence.of(unterminated));
        check(
                open.content().size() == 1 && open.content().get(0) instanceof Part.ToolCall,
                "finish() closes an open call body: " + open.content());

        // a payload that is not a JSON object is no call - and never content either
        int[] malformed = {1, 12, 13, 14, 6, 15, 2, 18, 5};
        Message none = ReplyParser.parse(new HarmonyReplyParser(TOK), IntSequence.of(malformed));
        check(none.content().isEmpty(), "malformed payload is no call: " + none.content());
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
            out.put(W[tokens.intAt(tokenStartIndex)].getBytes(StandardCharsets.UTF_8));
            return 1;
        }
    }

    private static final class FakeVocabulary implements Vocabulary {
        @Override
        public int size() {
            return W.length;
        }

        @Override
        public String token(int id) {
            if (!contains(id)) throw new NoSuchElementException("id " + id);
            return W[id];
        }

        @Override
        public int id(String text) {
            for (int i = 0; i < W.length; i++) if (W[i].equals(text)) return i;
            throw new NoSuchElementException(text);
        }

        @Override
        public boolean contains(int id) {
            return id >= 0 && id < W.length;
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
