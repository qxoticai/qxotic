package com.qxotic.jinfer.models.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.llm.Tokenizers;
import com.qxotic.jinfer.testkit.ModelFixture;
import com.qxotic.toknroll.Tokenizer;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * The strengthened wire law for the JSON-envelope pair: each family's FORCED selection must admit
 * its own rendered call at every token - envelope bytes, an offered name, schema-canonical
 * arguments - and commit the parsed call with the span's verbatim ids. Plus the no-think seeded
 * walk: the prompt-closed think pair must land the parser in content mode, not end it.
 */
final class JsonEnvelopeReplyLanguageTest {

    static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    "{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\","
                            + "\"properties\":{\"city\":{\"type\":\"string\"}},"
                            + "\"required\":[\"city\"]}}");

    static Tokenizer tokenizer(Path model) throws Exception {
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model);
        GGUF gguf;
        try (FileChannel ch = FileChannel.open(model, StandardOpenOption.READ)) {
            gguf = ModelLoader.readGguf(ch, model.toString());
        }
        return Tokenizers.fromGGUF(gguf);
    }

    /** Drives the forced walk through its own canonical wire, mask-checked at every token. */
    static void forcedWireLaw(Tokenizer tok, ChatTemplate template, String close) {
        ReplyLanguage.Selection sel =
                ReplyLanguage.Selection.of(
                        template.forcedCallLanguage(List.of(WEATHER)).orElseThrow(), tok);
        int[] prefix = sel.forcedPrefix();
        assertTrue(prefix.length > 2, "the envelope header is forced into the prompt");
        ReplyLanguage.Walk walk = sel.walk();
        for (int t : prefix) walk.feed(t);

        F32FloatTensor logits = F32FloatTensor.allocate(Arena.ofAuto(), tok.vocabulary().size());
        for (int t : tok.encode("{\"city\": \"Paris\"}").toArray()) {
            for (int i = 0; i < tok.vocabulary().size(); i++) logits.setFloat(i, 0f);
            assertTrue(walk.maskLogits(logits), "the walk must stay live through its own wire");
            assertTrue(logits.getFloat(t) == 0f, "the schema must admit canonical args token " + t);
            walk.feed(t);
        }
        // the trailing envelope bytes and the closing mark are forced by the mask, one path
        for (int t : tok.encode("}\n").toArray()) walk.feed(t);
        walk.feed(com.qxotic.jinfer.llm.SpecialTokens.require(tok, close));
        assertTrue(walk.accepted(), "the closed span is an acceptable end");
        Message m = walk.finish();
        Part.ToolCall c = (Part.ToolCall) m.content().get(0);
        assertEquals("get_weather", c.name());
        assertEquals(Map.of("city", "Paris"), c.arguments());
        assertTrue(
                c.verbatim() != null && c.verbatim().length() > 0,
                "a span-shaped call carries the payload's verbatim ids");
    }

    @Test
    void smolLm3ForcedSelectionAdmitsItsOwnWire() throws Exception {
        Tokenizer tok = tokenizer(ModelFixture.SMOLLM3_Q4.path());
        forcedWireLaw(tok, new SmolLm3ChatTemplate(tok, "01 January 2026"), "</tool_call>");
    }

    @Test
    void graniteForcedSelectionAdmitsItsOwnWire() throws Exception {
        Tokenizer tok = tokenizer(ModelFixture.GRANITE_41_3B_Q8.path());
        forcedWireLaw(tok, new GraniteTurnTemplate(tok), "</tool_call>");
    }

    @Test
    void mistralForcedSelectionAdmitsItsOwnWire() throws Exception {
        // the family that NEVER had a working pin: the whole header is forced, the args
        // schema-bound - the historical empty-prefix dead-end is structurally unexpressible
        Tokenizer tok = tokenizer(ModelFixture.MINISTRAL_3B_Q8.path());
        MistralChatTemplate template = new MistralChatTemplate(tok);
        ReplyLanguage.Selection sel =
                ReplyLanguage.Selection.of(
                        template.forcedCallLanguage(List.of(WEATHER)).orElseThrow(), tok);
        int[] prefix = sel.forcedPrefix();
        assertTrue(prefix.length >= 3, "[TOOL_CALLS] + name + [ARGS] all forced");
        ReplyLanguage.Walk walk = sel.walk();
        for (int t : prefix) walk.feed(t);
        F32FloatTensor logits = F32FloatTensor.allocate(Arena.ofAuto(), tok.vocabulary().size());
        for (int t : tok.encode("{\"city\": \"Paris\"}").toArray()) {
            for (int i = 0; i < tok.vocabulary().size(); i++) logits.setFloat(i, 0f);
            assertTrue(walk.maskLogits(logits));
            assertTrue(logits.getFloat(t) == 0f, "schema admits canonical args token " + t);
            walk.feed(t);
        }
        assertTrue(walk.accepted(), "the balanced close-less payload is an acceptable end");
        Message m = walk.finish();
        Part.ToolCall c = (Part.ToolCall) m.content().get(0);
        assertEquals("get_weather", c.name());
        assertEquals(Map.of("city", "Paris"), c.arguments());
    }

    static final Tool REFRESH =
            new Tool("refresh_cache", "{\"name\":\"refresh_cache\",\"parameters\":{}}");

    @Test
    void aMultiToolEnvelopeSelectionCommitsTheSecondTool() throws Exception {
        // both envelopes share the <tool_call> opener: candidacy holds them until the names
        // diverge inside the envelope bytes, and the second tool's wire commits cleanly
        Tokenizer tok = tokenizer(ModelFixture.SMOLLM3_Q4.path());
        SmolLm3ChatTemplate template = new SmolLm3ChatTemplate(tok, "01 January 2026");
        ReplyLanguage.Selection sel =
                ReplyLanguage.Selection.of(
                        template.forcedCallLanguage(List.of(WEATHER, REFRESH)).orElseThrow(), tok);
        ReplyLanguage.Walk walk = sel.walk();
        walk.feed(com.qxotic.jinfer.llm.SpecialTokens.require(tok, "<tool_call>"));
        for (int t : tok.encode("\n{\"name\": \"refresh_cache\", \"arguments\": {}}\n").toArray()) {
            walk.feed(t);
        }
        walk.feed(com.qxotic.jinfer.llm.SpecialTokens.require(tok, "</tool_call>"));
        assertTrue(walk.accepted());
        Part.ToolCall c = (Part.ToolCall) walk.finish().content().get(0);
        assertEquals("refresh_cache", c.name());
    }

    @Test
    void anInventedArgumentKeyIsUnsamplableInTheEnvelope() throws Exception {
        Tokenizer tok = tokenizer(ModelFixture.SMOLLM3_Q4.path());
        SmolLm3ChatTemplate template = new SmolLm3ChatTemplate(tok, "01 January 2026");
        ReplyLanguage.Selection sel =
                ReplyLanguage.Selection.of(
                        template.forcedCallLanguage(List.of(WEATHER)).orElseThrow(), tok);
        ReplyLanguage.Walk walk = sel.walk();
        for (int t : sel.forcedPrefix()) walk.feed(t);
        for (int t : tok.encode("{\"").toArray()) walk.feed(t);
        F32FloatTensor logits = F32FloatTensor.allocate(Arena.ofAuto(), tok.vocabulary().size());
        for (int i = 0; i < tok.vocabulary().size(); i++) logits.setFloat(i, 0f);
        assertTrue(walk.maskLogits(logits));
        assertTrue(logits.getFloat(tok.encode("city").toArray()[0]) == 0f);
        assertTrue(
                logits.getFloat(tok.encode("toolbench").toArray()[0]) == Float.NEGATIVE_INFINITY,
                "an invented key's first token is unsamplable");
    }

    @Test
    void theNoThinkSeededWalkLandsInContentMode() throws Exception {
        Tokenizer tok = tokenizer(ModelFixture.SMOLLM3_Q4.path());
        SmolLm3ChatTemplate template = new SmolLm3ChatTemplate(tok, "01 January 2026");
        var walk = (ReplyLanguage.Walk) template.parser();
        for (int t : template.replySeed(false)) walk.feed(t);
        assertFalse(walk.ended(), "the prompt-closed think pair is part of the language");
        walk.feed(tok.encode("Hello").toArray()[0]);
        assertFalse(walk.reasoning(), "post-seed text is CONTENT: the closed pair ended thinking");
        assertEquals("content", walk.pendingChannel());
    }
}
