package com.qxotic.jinfer.testkit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.llm.Tokenizers;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * The model-agnostic TOOL WIRE contract, tokenizer-only (no weights, no generation): each family
 * subclass states how its model EMITS a call - the generated wire, independent of the encode
 * implementation - and this battery proves the family's {@link ChatTemplate#parser} recovers it
 * structurally: non-trivial argument round-trips (unicode, newlines, nesting, numerics, empty),
 * multiple calls per reply, reasoning-then-call routing, content/call separation, malformed
 * payloads, verbatim payload ids, and the forced-call recipe's structural laws ({@code callSeed} is
 * a PREFIX of the generated wire; {@code callGrammar}'s pin ADMITS the family's own bytes).
 *
 * <p>Loads only the GGUF's tokenizer (assume-skips when the file is absent); runs in seconds.
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public abstract class AbstractToolWireTest {

    /** The GGUF whose tokenizer drives this family. */
    protected abstract Path modelPath();

    /** The family's native codec over that tokenizer. */
    protected abstract ChatTemplate template(Tokenizer tokenizer);

    /**
     * The family's GENERATED call wire: write {@code name(args)} exactly as the model emits it -
     * trusted marker ids and plain payload text. This fixture is deliberately independent of the
     * encode side; it documents the reply grammar the parser must recover.
     */
    protected abstract void call(TokenRuns runs, String name, Map<String, Object> args);

    /** The family's reasoning wire; default = the {@code <think>} span markers. */
    protected void think(TokenRuns runs, String text) {
        runs.id(SpecialTokens.require(tokenizer, "<think>"))
                .text(text)
                .id(SpecialTokens.require(tokenizer, "</think>"));
    }

    /** Plain answer content in the reply stream; families with channel framing override. */
    protected void content(TokenRuns runs, String text) {
        runs.text(text);
    }

    /** Anything the reply stream needs after its last call (Mistral's span-closing eos). */
    protected void endReply(TokenRuns runs) {}

    /** Garbage INSIDE the family's call framing - must parse to no call, never crash. */
    protected abstract void malformedCall(TokenRuns runs);

    /** How an argument VALUE survives the family's wire (MiniCPM's untyped wire stringifies). */
    protected Object expectedArg(Object value) {
        return value;
    }

    /** Whether this family's reply grammar has a reasoning channel to test. */
    protected boolean hasThinkWire() {
        return SpecialTokens.find(tokenizer, "<think>").isPresent();
    }

    /** Whether one reply can carry several calls (Harmony stops at {@code <|call|>}: one). */
    protected boolean supportsMultipleCalls() {
        return true;
    }

    protected Tokenizer tokenizer;
    protected ChatTemplate template;

    @BeforeAll
    void load() throws Exception {
        Path model = modelPath();
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model);
        GGUF gguf;
        try (FileChannel ch = FileChannel.open(model, StandardOpenOption.READ)) {
            gguf = ModelLoader.readGguf(ch, model.toString());
        }
        tokenizer = Tokenizers.fromGGUF(gguf);
        template = template(tokenizer);
    }

    // ---- helpers ----

    protected TokenRuns runs() {
        return new TokenRuns(tokenizer);
    }

    protected Message parse(TokenRuns reply) {
        return ReplyParser.parse(template.parser(), reply.build());
    }

    protected static List<Part.ToolCall> calls(Message m) {
        return m.content().stream()
                .filter(p -> p instanceof Part.ToolCall)
                .map(p -> (Part.ToolCall) p)
                .toList();
    }

    protected Map<String, Object> expected(Map<String, Object> args) {
        Map<String, Object> out = new LinkedHashMap<>();
        for (var e : args.entrySet()) out.put(e.getKey(), expectedArg(e.getValue()));
        return out;
    }

    private void assertRoundTrip(String name, Map<String, Object> args) {
        TokenRuns reply = runs();
        call(reply, name, args);
        endReply(reply);
        Message m = parse(reply);
        List<Part.ToolCall> parsed = calls(m);
        assertEquals(1, parsed.size(), () -> "expected one call: " + m.content());
        assertEquals(name, parsed.get(0).name());
        assertEquals(expected(args), parsed.get(0).arguments());
    }

    // ---- the battery ----

    @Test
    void simpleCall() {
        assertRoundTrip("get_weather", Map.of("city", "Paris"));
    }

    @Test
    void unicodeAndQuotesInArguments() {
        assertRoundTrip("send_message", Map.of("text", "He said \"grüß dich\" 🌊 <ok>&done"));
    }

    @Test
    void multilineArgument() {
        assertRoundTrip("send_message", Map.of("text", "line one\nline two\n\tindented"));
    }

    @Test
    void numericAndBooleanArguments() {
        var args = new LinkedHashMap<String, Object>();
        args.put("count", 3L);
        args.put("ratio", 2.5);
        args.put("flag", Boolean.TRUE);
        assertRoundTrip("configure", args);
    }

    @Test
    void nestedArguments() {
        var filters = new LinkedHashMap<String, Object>();
        filters.put("stars", 4L);
        filters.put("tags", List.of("pool", "spa"));
        var args = new LinkedHashMap<String, Object>();
        args.put("city", "Rome");
        args.put("filters", filters);
        assertRoundTrip("search_hotels", args);
    }

    @Test
    void emptyArguments() {
        TokenRuns reply = runs();
        call(reply, "refresh_cache", Map.of());
        endReply(reply);
        List<Part.ToolCall> parsed = calls(parse(reply));
        assertEquals(1, parsed.size());
        assertEquals("refresh_cache", parsed.get(0).name());
        assertTrue(parsed.get(0).arguments().isEmpty(), String.valueOf(parsed.get(0)));
    }

    @Test
    void multipleCallsInOneReply() {
        Assumptions.assumeTrue(supportsMultipleCalls(), "one call per reply in this family");
        TokenRuns reply = runs();
        call(reply, "get_weather", Map.of("city", "Paris"));
        call(reply, "get_time", Map.of("city", "Tokyo"));
        endReply(reply);
        List<Part.ToolCall> parsed = calls(parse(reply));
        assertEquals(2, parsed.size(), () -> "expected two calls: " + parsed);
        assertEquals("get_weather", parsed.get(0).name());
        assertEquals("get_time", parsed.get(1).name());
        assertEquals(expected(Map.of("city", "Tokyo")), parsed.get(1).arguments());
    }

    @Test
    void reasoningThenCall() {
        Assumptions.assumeTrue(hasThinkWire(), "family has no reasoning wire");
        TokenRuns reply = runs();
        think(reply, "I should call the weather tool.");
        call(reply, "get_weather", Map.of("city", "Paris"));
        endReply(reply);
        Message m = parse(reply);
        Part.Reasoning reasoning = m.reasoning();
        assertTrue(
                reasoning != null && reasoning.text().contains("weather tool"),
                () -> "reasoning preserved: " + m.content());
        assertEquals(1, calls(m).size(), () -> String.valueOf(m.content()));
        assertTrue(
                !m.text().contains("weather tool"),
                () -> "reasoning must not leak into content: " + m.text());
    }

    @Test
    void contentAndCallSeparate() {
        TokenRuns reply = runs();
        content(reply, "Let me check that for you.");
        call(reply, "get_weather", Map.of("city", "Paris"));
        endReply(reply);
        Message m = parse(reply);
        assertTrue(m.text().contains("Let me check"), () -> "content kept: " + m.text());
        assertTrue(
                !m.text().contains("Paris") && !m.text().contains("city"),
                () -> "payload must not leak into content: " + m.text());
        assertEquals(1, calls(m).size());
    }

    @Test
    void malformedPayloadIsNoCall() {
        TokenRuns reply = runs();
        malformedCall(reply);
        endReply(reply);
        assertEquals(0, calls(parse(reply)).size());
    }

    @Test
    void verbatimIdsCoverThePayload() {
        TokenRuns reply = runs();
        call(reply, "get_weather", Map.of("city", "Paris"));
        endReply(reply);
        List<Part.ToolCall> parsed = calls(parse(reply));
        assertEquals(1, parsed.size());
        IntSequence verbatim = parsed.get(0).verbatim();
        assertTrue(verbatim != null && verbatim.length() > 0, "calls carry verbatim payload ids");
    }

    @Test
    void channelsClassifyTheWire() {
        // the channel law behind channel-scoped grammars: reasoning and call payloads are never
        // output channels; content is. Fed segment by segment so the wire builders' boundaries
        // give us the ground truth for free.
        ReplyParser parser = template.parser();
        java.util.Set<String> out = parser.outputChannels();
        assertTrue(!out.isEmpty(), "every parser declares its output channels");
        for (int t : template.replySeed(hasThinkWire())) parser.feed(t);
        if (hasThinkWire()) {
            TokenRuns thinkRun = runs();
            think(thinkRun, "weighing the options here");
            int[] wire = thinkRun.build().toArray();
            boolean sawReasoning = false;
            for (int i = 0; i < wire.length; i++) {
                String ch = parser.pendingChannel();
                if (ch != null && !out.contains(ch)) sawReasoning = true;
                // position 0 is the open marker itself: pending is legitimately still the
                // content channel there (the model CHOOSING to reason from content)
                assertTrue(
                        i == 0 || ch == null || !out.contains(ch),
                        "inside the think wire, pending must never be an output channel: " + ch);
                parser.feed(wire[i]);
            }
            assertTrue(sawReasoning, "the think wire must classify as a non-output channel");
        }
        TokenRuns contentRun = runs();
        content(contentRun, "The answer is forty-two.");
        boolean sawOutput = false;
        for (int t : contentRun.build().toArray()) {
            String ch = parser.pendingChannel();
            if (ch != null && out.contains(ch)) sawOutput = true;
            parser.feed(t);
        }
        assertTrue(sawOutput, "content wire must reach an output channel");
        // the call wire must pass through a claimed region (payload or structure) - a grammar
        // over the output channels can never touch call syntax
        TokenRuns callRun = runs();
        call(callRun, "get_weather", Map.of("city", "Paris"));
        int[] callWire = callRun.build().toArray();
        boolean sawClaimed = false;
        for (int i = 0; i < callWire.length; i++) {
            if (i > 0) { // position 0 may legitimately still be the content channel (pre-marker)
                String ch = parser.pendingChannel();
                if (ch == null || !out.contains(ch)) sawClaimed = true;
            }
            parser.feed(callWire[i]);
        }
        assertTrue(sawClaimed, "the call wire must be claimed away from the output channels");
    }

    // ---- forced-call recipe laws (seed / pin / epilogue vs the family's OWN wire) ----

    @Test
    void callSeedIsAPrefixOfTheGeneratedWire() {
        int[] seed = template.callSeed();
        Assumptions.assumeTrue(seed.length > 0, "family declares no call seed");
        TokenRuns reply = runs();
        call(reply, "get_weather", Map.of("city", "Paris"));
        int[] wire = reply.build().toArray();
        assertTrue(wire.length >= seed.length, "wire shorter than seed");
        for (int i = 0; i < seed.length; i++) {
            assertEquals(seed[i], wire[i], "seed diverges from the generated wire at " + i);
        }
    }

    @Test
    void callGrammarAdmitsTheFamilysOwnWire() {
        int[] seed = template.callSeed();
        var pin =
                template.callGrammar(
                        List.of(
                                new com.qxotic.jinfer.chat.Tool("get_weather", "{\"x\":1}"),
                                new com.qxotic.jinfer.chat.Tool("get_time", "{\"x\":1}")));
        Assumptions.assumeTrue(pin.isPresent(), "family declares no call grammar");
        TokenRuns reply = runs();
        call(reply, "get_weather", Map.of("city", "Paris"));
        int[] wire = reply.build().toArray();
        Grammar.Cursor cursor = Grammar.of(pin.get(), tokenizer).cursor();
        // walk the generated wire from just after the seed; every token must keep the matcher
        // alive (something still admissible) until the pin is fully matched - the released
        // region after that is the model's own
        var probe =
                com.qxotic.jinfer.F32FloatTensor.allocate(
                        java.lang.foreign.Arena.ofAuto(), tokenizer.vocabulary().size());
        int i = seed.length;
        while (!cursor.exhausted()) {
            assertTrue(i < wire.length, "wire ended before the pin was satisfied");
            cursor.advanceWith(wire[i]);
            final int at = i;
            if (!cursor.exhausted()) {
                assertTrue(
                        cursor.maskLogits(probe),
                        () -> "pin rejects the family's own wire at token " + at);
            }
            i++;
        }
    }
}
