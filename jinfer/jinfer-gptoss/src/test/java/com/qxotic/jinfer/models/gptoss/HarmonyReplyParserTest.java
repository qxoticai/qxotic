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

    // ids:           0            1             2          3           4            5
    static final String[] W = {
        "<|start|>",
        "<|channel|>",
        "<|message|>",
        "<|end|>",
        "<|return|>",
        "assistant",
        //   6           7          8            9
        "analysis",
        "final",
        "thinking...",
        "The answer is 4."
    };
    static final int SPECIALS = 5; // ids 0..4 special; 5.. plain

    static final Tokenizer TOK = new FakeTokenizer();

    static void check(boolean ok, String what) {
        Assertions.assertTrue(ok, what);
    }

    record Step(String fragment, boolean reasoning) {}

    @Test
    void harmonyChannelRouting() {
        // <|channel|>analysis<|message|>thinking...<|end|><|start|>assistant<|channel|>final
        // <|message|>The answer is 4.<|return|>
        int[] reply = {1, 6, 2, 8, 3, 0, 5, 1, 7, 2, 9, 4};
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
                        && t.verbatim().toList().equals(List.of(9)),
                "final body carries verbatim ids");

        // header text (role, channel names) never leaks into either channel
        String all = m.text() + m.content();
        check(!all.contains("assistant") && !all.contains("analysis"), "header text never leaks");

        // one-shot equals streamed
        Message oneshot = ReplyParser.parse(new HarmonyReplyParser(TOK), IntSequence.of(reply));
        check(oneshot.equals(m), "streamed == one-shot");
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
