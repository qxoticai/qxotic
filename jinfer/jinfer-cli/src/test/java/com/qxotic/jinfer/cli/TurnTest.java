package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Channel;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The terminal rendering of a reply, without a model: {@code --stream false} must show exactly what
 * {@code --stream true} shows, later. Reasoning goes where {@code --think} sends it and never into
 * the answer.
 */
class TurnTest {

    private record Output(String out, String err) {}

    @ParameterizedTest(name = "--think {0}")
    @ValueSource(strings = {"on", "inline", "off"})
    void nonStreamingIsStreamingReplayed(String think) {
        List<ChatEngine.Delta> reply =
                List.of(reasoning("why "), content("42"), reasoning("more "), content("!"));
        Output streamed = run(options(true, think), reply);
        Output buffered = run(options(false, think), reply);
        assertEquals(streamed.out(), buffered.out(), "stdout");
        assertEquals(streamed.err(), buffered.err(), "stderr");
    }

    @Test
    void reasoningNeverLandsInTheAnswer() {
        Output o = run(options(false, "on"), List.of(reasoning("why "), content("42")));
        assertEquals("42\n", o.out(), "stdout is the answer, one line");
        assertTrue(o.err().contains("[Start thinking]\nwhy \n[End thinking]"), o.err());
        assertFalse(o.err().contains("42"));

        Output inline = run(options(false, "inline"), List.of(reasoning("why "), content("42")));
        assertEquals("[Start thinking]\nwhy \n[End thinking]\n\n42\n", inline.out());
    }

    @Test
    void anEmptyReplyIsOneLineInBothModes() {
        assertEquals("\n", run(options(true, "on"), List.of()).out());
        assertEquals("\n", run(options(false, "on"), List.of()).out());
    }

    // ---- harness ----

    private static Output run(Options options, List<ChatEngine.Delta> deltas) {
        PrintStream realOut = System.out, realErr = System.err;
        ByteArrayOutputStream out = new ByteArrayOutputStream(), err = new ByteArrayOutputStream();
        try {
            System.setOut(new PrintStream(out, true, StandardCharsets.UTF_8));
            System.setErr(new PrintStream(err, true, StandardCharsets.UTF_8));
            Turn turn = new Turn(NEVER_CALLED, options, false);
            deltas.forEach(turn::on);
            turn.finish(
                    new ChatEngine.Completion(
                            null, null, true, 0, 0, PromptCache.Tier.SESSION, null),
                    4096);
        } finally {
            System.setOut(realOut);
            System.setErr(realErr);
        }
        return new Output(
                out.toString(StandardCharsets.UTF_8), err.toString(StandardCharsets.UTF_8));
    }

    private static ChatEngine.Delta reasoning(String text) {
        return new ChatEngine.Delta(Channel.REASONING, text, IntSequence.empty());
    }

    private static ChatEngine.Delta content(String text) {
        return new ChatEngine.Delta(Channel.CONTENT, text, IntSequence.empty());
    }

    private static Options options(boolean stream, String think) {
        boolean on = !think.equals("off"), inline = think.equals("inline");
        return new Options(
                Path.of("model.gguf"),
                null,
                null,
                "hi",
                null,
                false,
                null,
                null,
                null,
                null,
                null,
                128,
                null,
                stream,
                false,
                on,
                inline,
                false,
                false,
                null,
                false,
                4);
    }

    /** Turn only consults the tokenizer for --echo and the raw lane; neither runs here. */
    private static final Tokenizer NEVER_CALLED =
            new Tokenizer() {
                @Override
                public Vocabulary vocabulary() {
                    throw new AssertionError("tokenizer consulted");
                }

                @Override
                public void encodeInto(
                        CharSequence text, int start, int end, IntSequence.Builder out) {
                    throw new AssertionError("tokenizer consulted");
                }

                @Override
                public int countTokens(CharSequence text, int start, int end) {
                    throw new AssertionError("tokenizer consulted");
                }

                @Override
                public int decodeBytesInto(IntSequence tokens, int index, ByteBuffer out) {
                    throw new AssertionError("tokenizer consulted");
                }
            };
}
