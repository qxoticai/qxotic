package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Channel;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Role;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.testkit.TestLanguageModel;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.StandardTokenType;
import com.qxotic.toknroll.TokenType;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The terminal rendering of a reply, without a model: {@code --no-stream} must show exactly what
 * {@code --stream} shows, later. Reasoning goes where {@code --think} sends it and never into the
 * answer.
 */
class TurnTest {

    private record Output(String out, String err) {}

    /** A rate over a handful of tokens is fixed latency; those report the count and the time. */
    @Test
    void ratesNeedEnoughTokensToMeanThroughput() {
        assertEquals("1 token in 0.25 s", Turn.speed(1, 250_000_000L));
        assertEquals("17 tokens in 0.73 s", Turn.speed(17, 730_000_000L));
        assertEquals("0 tokens in 0.00 s", Turn.speed(0, 0));
        assertEquals("64.00 tokens/s (32)", Turn.speed(Turn.MIN_RATE_TOKENS, 500_000_000L));
    }

    @ParameterizedTest(name = "--think {0}")
    @ValueSource(strings = {"on", "inline", "off"})
    void nonStreamingIsStreamingReplayed(String think) {
        List<ChatEngine.Delta> reply =
                List.of(reasoning("why "), content("42"), reasoning("more "), content("!"));
        Output streamed = run(options(true, think), reply);
        Output buffered = run(options(false, think), reply);
        assertEquals(streamed.out(), buffered.out(), "stdout");
        assertEquals("Generating response ...\n" + streamed.err(), buffered.err(), "stderr");
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
    void aReplyCutByTheContextSaysSoAndNamesTheFlag() {
        // a reasoning loop ran into the context wall: stdout shows no answer, so stderr says why
        Output o =
                finished(
                        options(true, "inline"),
                        List.of(reasoning("10 * 3 = 30. ")),
                        completion(Generator.FinishReason.LENGTH, 54, 0),
                        64,
                        -1);
        assertTrue(
                o.err()
                        .startsWith(
                                "stopped: the context is full (64 tokens) during the reasoning, no"
                                        + " answer; raise --context-capacity\ncontext: 64/64 "),
                o.err());
    }

    @Test
    void aReplyCutByTheOutputBudgetNamesThatFlagInstead() {
        Options options =
                Options.parse("instruct", "-m", "unused", "hi", "-n", "10", "--color", "off");
        Output o =
                finished(
                        options,
                        List.of(reasoning("hmm "), content("The answer is")),
                        completion(Generator.FinishReason.LENGTH, 10, 0),
                        4096,
                        -1);
        assertTrue(
                o.err().contains("[End thinking]\n\nstopped: --max-output-tokens 10 reached\n"),
                o.err());
        assertFalse(o.err().contains("no answer"), "the reply did answer, if briefly");
    }

    @Test
    void aReasoningCapThatFiredIsVisible() {
        // the cap closes the span with a bare paragraph break by default: nothing in the text
        // says the model did not finish its thought
        Output capped =
                finished(
                        options(true, "on"),
                        List.of(reasoning("1.  **Analyze"), content("Blue.")),
                        completion(Generator.FinishReason.STOP, 8, 6),
                        4096,
                        6);
        assertTrue(
                capped.err()
                        .contains(
                                "reasoning: cut at its cap of 6 tokens; raise"
                                        + " --max-reasoning-tokens (-1: uncapped)\n"),
                capped.err());

        Output finished =
                finished(
                        options(true, "on"),
                        List.of(reasoning("done"), content("Blue.")),
                        completion(Generator.FinishReason.STOP, 8, 5),
                        4096,
                        6);
        assertFalse(finished.err().contains("reasoning: cut"), finished.err());
        assertFalse(finished.err().contains("stopped"), finished.err());
    }

    @Test
    void echoEndsItsLineBeforeTheSummary() {
        // --echo leaves a run of token spellings on stderr; the summary must not join it
        Options options =
                Options.parse("instruct", "-m", "unused", "hi", "--echo", "--color", "off");
        Output o =
                finished(
                        options,
                        List.of(content("Hello")),
                        completion(Generator.FinishReason.STOP, 3, 0),
                        4096,
                        -1);
        assertTrue(o.err().startsWith("\ncontext: "), o.err());
    }

    @Test
    void anEmptyReplyIsOneLineInBothModes() {
        assertEquals("\n", run(options(true, "on"), List.of()).out());
        assertEquals("\n", run(options(false, "on"), List.of()).out());
    }

    @Test
    void brokenStdoutStopsGenerationRatherThanSilentlyDroppingTokens() {
        var broken =
                new PrintStream(
                        new java.io.OutputStream() {
                            public void write(int value) throws java.io.IOException {
                                throw new java.io.IOException("closed pipe");
                            }
                        });
        var io =
                new Main.IO(
                        java.io.InputStream.nullInputStream(),
                        broken,
                        new PrintStream(java.io.OutputStream.nullOutputStream()));
        var turn = new Turn(NEVER_CALLED, options(true, "off"), false, io);
        assertThrows(java.io.UncheckedIOException.class, () -> turn.on(content("answer")));
        var inline = new Turn(NEVER_CALLED, options(true, "inline"), false, io);
        assertThrows(java.io.UncheckedIOException.class, () -> inline.on(reasoning("thought")));
    }

    @Test
    void failedGenerationRestoresReasoningFraming() {
        var capture = new CliFixtures.Capture("");
        Options o = Options.parse("instruct", "-m", "unused", "hi", "--color", "on");
        var turn = new Turn(NEVER_CALLED, o, false, capture.io);
        assertThrows(
                IllegalStateException.class,
                () -> {
                    try (turn) {
                        turn.on(reasoning("unfinished thought"));
                        throw new IllegalStateException("test failure");
                    }
                });
        assertTrue(capture.err().contains("[End thinking]"));
        assertTrue(capture.err().replace("\r\n", "\n").endsWith("\033[0m\n"));
        String closed = capture.err();
        turn.close();
        assertEquals(closed, capture.err(), "closing twice must not print another footer");
    }

    @Test
    void rawOutputFiltersControlTokensAndEchoEscapesTerminalControls() {
        var capture = new CliFixtures.Capture("");
        Options o =
                Options.parse(
                        "instruct",
                        "-m",
                        "unused",
                        "hello",
                        "--raw-prompt",
                        "--echo",
                        "--color",
                        "off");
        try (var turn =
                Turn.startRaw(controlTokenizer(), new int[] {27, 10, 9, 'a'}, o, capture.io)) {
            turn.on(new ChatEngine.Delta(Channel.CONTENT, "<stop>", IntSequence.of(0)));
            turn.on(new ChatEngine.Delta(Channel.CONTENT, "x", IntSequence.of('x')));
            turn.finish(
                    new ChatEngine.Completion(
                            null, null, true, 0, 0, PromptCache.Tier.SESSION, null),
                    4096,
                    -1);
        }
        assertEquals("x\n", capture.out().replace("\r\n", "\n"));
        assertTrue(capture.err().startsWith("\\u001b\n\\u0009a"), capture.err());
        assertTrue(capture.err().contains("\\u0000"));
        assertFalse(capture.err().contains("\033"));
    }

    /**
     * The echo already shows every reasoning token on stderr; a second copy interleaves with it.
     */
    @Test
    void echoedThoughtsAreNotRenderedAgain() {
        var capture = new CliFixtures.Capture("");
        Options o =
                Options.parse(
                        "instruct",
                        "-m",
                        "unused",
                        "hello",
                        "--raw-prompt",
                        "--echo",
                        "--color",
                        "off");
        try (var turn = Turn.startRaw(controlTokenizer(), new int[] {'a'}, o, capture.io)) {
            turn.on(new ChatEngine.Delta(Channel.REASONING, "y", IntSequence.of('y')));
            turn.on(new ChatEngine.Delta(Channel.CONTENT, "x", IntSequence.of('x')));
            turn.finish(
                    new ChatEngine.Completion(
                            null, null, true, 0, 0, PromptCache.Tier.SESSION, null),
                    4096,
                    -1);
        }
        assertEquals("ayx", capture.err(), "the echo alone: prompt, thought, answer");
        assertEquals("x\n", capture.out().replace("\r\n", "\n"), "a redirected stdout keeps it");
    }

    private static Tokenizer controlTokenizer() {
        Tokenizer base = TestLanguageModel.TOKENIZER;
        Vocabulary words = base.vocabulary();
        Vocabulary vocabulary =
                new Vocabulary() {
                    public int size() {
                        return words.size();
                    }

                    public String token(int id) {
                        return words.token(id);
                    }

                    public int id(String text) {
                        return words.id(text);
                    }

                    public boolean contains(int id) {
                        return words.contains(id);
                    }

                    public boolean contains(String text) {
                        return words.contains(text);
                    }

                    public Iterator<Map.Entry<String, Integer>> iterator() {
                        return words.iterator();
                    }

                    public boolean isTokenOfType(int id, TokenType type) {
                        return id == 0
                                ? type == StandardTokenType.CONTROL
                                : words.isTokenOfType(id, type);
                    }
                };
        return new Tokenizer() {
            public Vocabulary vocabulary() {
                return vocabulary;
            }

            public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
                base.encodeInto(text, start, end, out);
            }

            public int countTokens(CharSequence text, int start, int end) {
                return base.countTokens(text, start, end);
            }

            public int decodeBytesInto(IntSequence tokens, int from, ByteBuffer out) {
                return base.decodeBytesInto(tokens, from, out);
            }
        };
    }

    // ---- harness ----

    private static Output run(Options options, List<ChatEngine.Delta> deltas) {
        return finished(
                options,
                deltas,
                new ChatEngine.Completion(null, null, true, 0, 0, PromptCache.Tier.SESSION, null),
                4096,
                -1);
    }

    private static Output finished(
            Options options,
            List<ChatEngine.Delta> deltas,
            ChatEngine.Completion completion,
            int contextCapacity,
            int reasoningCap) {
        ByteArrayOutputStream out = new ByteArrayOutputStream(), err = new ByteArrayOutputStream();
        Main.IO io =
                new Main.IO(
                        java.io.InputStream.nullInputStream(),
                        new PrintStream(out, true, StandardCharsets.UTF_8),
                        new PrintStream(err, true, StandardCharsets.UTF_8));
        Turn turn = new Turn(NEVER_CALLED, options, false, io);
        deltas.forEach(turn::on);
        turn.finish(completion, contextCapacity, reasoningCap);
        return new Output(
                out.toString(StandardCharsets.UTF_8).replace("\r\n", "\n"),
                err.toString(StandardCharsets.UTF_8).replace("\r\n", "\n"));
    }

    /** A 10-token prompt, {@code generated} tokens, {@code reasoning} of them in the think span. */
    private static ChatEngine.Completion completion(
            Generator.FinishReason finish, int generated, int reasoning) {
        int[] tokens = new int[generated];
        Message reply =
                new Message(
                        Role.ASSISTANT,
                        List.of(
                                new Content.Reasoning(
                                        List.of(new Content.Text("thought")),
                                        IntSequence.of(new int[reasoning])),
                                new Content.Text("answer")));
        return new ChatEngine.Completion(
                reply,
                new Generator.GenerationResult(
                        tokens, OptionalInt.empty(), finish, Duration.ZERO, Duration.ZERO),
                false,
                10,
                0,
                PromptCache.Tier.FRESH,
                null);
    }

    private static ChatEngine.Delta reasoning(String text) {
        return new ChatEngine.Delta(Channel.REASONING, text, IntSequence.empty());
    }

    private static ChatEngine.Delta content(String text) {
        return new ChatEngine.Delta(Channel.CONTENT, text, IntSequence.empty());
    }

    private static Options options(boolean stream, String think) {
        return Options.parse(
                "instruct",
                "-m",
                "unused",
                "hi",
                stream ? "--stream" : "--no-stream",
                "--think",
                think,
                "--color",
                "off");
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
