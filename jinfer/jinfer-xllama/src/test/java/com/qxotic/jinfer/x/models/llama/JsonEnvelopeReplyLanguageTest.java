package com.qxotic.jinfer.x.models.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.chat.ChatTemplate;
import com.qxotic.jinfer.x.chat.Content;
import com.qxotic.jinfer.x.chat.ReplyLanguage;
import com.qxotic.jinfer.x.chat.Tool;
import com.qxotic.jinfer.x.kernels.ModelLoader;
import com.qxotic.jinfer.x.llm.SpecialTokens;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Each forced-call language must admit and parse its own model-family wire format. */
@Tag("integration")
class JsonEnvelopeReplyLanguageTest {

    private static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    Map.of(
                            "name",
                            "get_weather",
                            "parameters",
                            Map.of(
                                    "type",
                                    "object",
                                    "properties",
                                    Map.of("city", Map.of("type", "string")),
                                    "required",
                                    List.of("city"))));

    @Test
    void smolLm3AdmitsItsForcedWire() throws Exception {
        Tokenizer tokenizer = tokenizer("hf.co/ggml-org/SmolLM3-3B-GGUF/SmolLM3-Q4_K_M.gguf");
        assertJsonEnvelope(tokenizer, new SmolLm3ChatTemplate(tokenizer), "</tool_call>");
    }

    @Test
    void graniteAdmitsItsForcedWire() throws Exception {
        Tokenizer tokenizer =
                tokenizer("hf.co/ibm-granite/granite-4.1-3b-GGUF/granite-4.1-3b-Q8_0.gguf");
        assertJsonEnvelope(tokenizer, new GraniteChatTemplate(tokenizer), "</tool_call>");
    }

    @Test
    void mistralAdmitsItsForcedWire() throws Exception {
        Tokenizer tokenizer = tokenizer("hf.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF:Q8_0");
        ReplyLanguage.Selection selection =
                new MistralChatTemplate(tokenizer).forcedCall(List.of(WEATHER)).orElseThrow();
        ReplyLanguage.Walk walk = selection.walk();
        feed(walk, selection.forcedPrefix());
        assertAdmitted(walk, tokenizer, "{\"city\": \"Paris\"}");
        assertTrue(walk.accepted());
        assertCall(walk);
    }

    @Test
    void smolLm3CommitsTheSelectedToolFromSeveralOffers() throws Exception {
        Tokenizer tokenizer = tokenizer("hf.co/ggml-org/SmolLM3-3B-GGUF/SmolLM3-Q4_K_M.gguf");
        Tool refresh = new Tool("refresh_cache", Map.of("parameters", Map.of()));
        ReplyLanguage.Walk walk =
                new SmolLm3ChatTemplate(tokenizer)
                        .forcedCall(List.of(WEATHER, refresh))
                        .orElseThrow()
                        .walk();
        walk.feed(SpecialTokens.require(tokenizer, "<tool_call>"));
        feed(
                walk,
                tokenizer.encode("\n{\"name\": \"refresh_cache\", \"arguments\": {}}\n").toArray());
        walk.feed(SpecialTokens.require(tokenizer, "</tool_call>"));
        assertTrue(walk.accepted());
        assertEquals("refresh_cache", ((Content.ToolCall) walk.finish().content().get(0)).name());
    }

    @Test
    void forcedSchemaRejectsAnUnknownArgument() throws Exception {
        Tokenizer tokenizer = tokenizer("hf.co/ggml-org/SmolLM3-3B-GGUF/SmolLM3-Q4_K_M.gguf");
        ReplyLanguage.Selection selection =
                new SmolLm3ChatTemplate(tokenizer).forcedCall(List.of(WEATHER)).orElseThrow();
        ReplyLanguage.Walk walk = selection.walk();
        feed(walk, selection.forcedPrefix());
        feed(walk, tokenizer.encode("{\"").toArray());

        MemoryView<MemorySegment> logits = zeros(tokenizer);
        assertTrue(walk.maskLogits(logits));
        int city = tokenizer.encode("city").toArray()[0];
        int invented = tokenizer.encode("toolbench").toArray()[0];
        assertEquals(0f, Views.getFloat(logits, city, "logits"));
        assertEquals(Float.NEGATIVE_INFINITY, Views.getFloat(logits, invented, "logits"));
    }

    private static void assertJsonEnvelope(
            Tokenizer tokenizer, ChatTemplate template, String close) {
        ReplyLanguage.Selection selection = template.forcedCall(List.of(WEATHER)).orElseThrow();
        assertTrue(selection.forcedPrefix().length > 2);
        ReplyLanguage.Walk walk = selection.walk();
        feed(walk, selection.forcedPrefix());
        assertAdmitted(walk, tokenizer, "{\"city\": \"Paris\"}");
        feed(walk, tokenizer.encode("}\n").toArray());
        walk.feed(SpecialTokens.require(tokenizer, close));
        assertTrue(walk.accepted());
        assertCall(walk);
    }

    private static void assertAdmitted(ReplyLanguage.Walk walk, Tokenizer tokenizer, String text) {
        for (int token : tokenizer.encode(text).toArray()) {
            MemoryView<MemorySegment> logits = zeros(tokenizer);
            assertTrue(walk.maskLogits(logits));
            assertEquals(0f, Views.getFloat(logits, token, "logits"));
            walk.feed(token);
        }
    }

    private static MemoryView<MemorySegment> zeros(Tokenizer tokenizer) {
        return Views.fromFloatArray(
                new PanamaMemoryArena(Arena.ofAuto()), new float[tokenizer.vocabulary().size()]);
    }

    private static void assertCall(ReplyLanguage.Walk walk) {
        Content.ToolCall call = (Content.ToolCall) walk.finish().content().get(0);
        assertEquals("get_weather", call.name());
        assertEquals(Map.of("city", "Paris"), call.arguments());
        assertTrue(!call.verbatim().isEmpty());
    }

    private static void feed(ReplyLanguage.Walk walk, int[] tokens) {
        for (int token : tokens) walk.feed(token);
    }

    private static Tokenizer tokenizer(String reference) throws Exception {
        Path path = TestModels.require(reference);
        try (FileChannel file = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(file, path.toString());
            return GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        }
    }
}
