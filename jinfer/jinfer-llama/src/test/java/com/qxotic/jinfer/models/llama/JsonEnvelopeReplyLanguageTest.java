package com.qxotic.jinfer.models.llama;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
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
    void granite42AdmitsItsForcedWire() throws Exception {
        Tokenizer tokenizer =
                tokenizer("hf.co/ibm-granite/granite-4.2-3b-GGUF/granite-4.2-3b-Q8_0.gguf");
        ReplyLanguage.Selection selection =
                new GraniteChatTemplate(tokenizer).forcedCall(List.of(WEATHER)).orElseThrow();
        ReplyLanguage.Walk walk = selection.walk();
        feed(walk, selection.forcedPrefix());
        assertAdmitted(walk, tokenizer, ">\n<parameter=city>\nParis\n</parameter>\n</function>\n");
        walk.feed(SpecialTokens.require(tokenizer, "</tool_call>"));
        assertTrue(walk.accepted());
        assertCall(walk);
    }

    @Test
    void mistralAdmitsItsForcedWire() throws Exception {
        Tokenizer tokenizer = tokenizer("hf.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF:Q8_0");
        MistralChatTemplate template = new MistralChatTemplate(tokenizer);
        assertEquals(
                IntSequence.of(SpecialTokens.require(tokenizer, "<s>")), template.promptStart());
        ReplyLanguage.Selection selection = template.forcedCall(List.of(WEATHER)).orElseThrow();
        ReplyLanguage.Walk walk = selection.walk();
        feed(walk, selection.forcedPrefix());
        assertAdmitted(walk, tokenizer, "{\"city\": \"Paris\"}");
        assertTrue(walk.accepted());
        Content.ToolCall call = (Content.ToolCall) walk.finish().content().get(0);
        assertEquals("get_weather", call.name());
        assertEquals(Map.of("city", "Paris"), call.arguments());
        // Mistral's wire has an INTERIOR mark ([ARGS]): not span-shaped, and with no free hole
        // there is no exact splice unit - verbatim is EMPTY by the verbatim law (a missing
        // splice merely re-tokenizes the echo; a wrong one would corrupt it)
        assertTrue(call.verbatim().isEmpty());
    }

    @Test
    void llama32AdmitsAndParsesItsBareJsonWire() throws Exception {
        Tokenizer tokenizer =
                tokenizer(
                        "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF/"
                                + "Llama-3.2-1B-Instruct-Q8_0.gguf");
        Llama32ChatTemplate template = new Llama32ChatTemplate(tokenizer);
        String wire = "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Paris\"}}";

        ReplyLanguage.Walk walk = template.forcedCall(List.of(WEATHER)).orElseThrow().walk();
        assertAdmitted(walk, tokenizer, wire);
        assertTrue(walk.accepted());
        assertCall(walk);

        Message parsed = ReplyParser.parse(template.parser(tokenizer), tokenizer.encode(wire));
        Content.ToolCall call = (Content.ToolCall) parsed.content().getFirst();
        assertEquals("get_weather", call.name());
        assertEquals(Map.of("city", "Paris"), call.arguments());

        ReplyParser unclaimed = template.parser(tokenizer);
        unclaimed.disableToolCalls();
        Message text = ReplyParser.parse(unclaimed, tokenizer.encode(wire));
        assertEquals(wire, text.text());
        assertTrue(text.content().stream().noneMatch(Content.ToolCall.class::isInstance));
        assertEquals(tokenizer.encode(wire), ((Content.Text) text.content().getFirst()).verbatim());

        ReplyParser seeded = template.parser(tokenizer);
        seeded.seed(tokenizer.encode(" \n"));
        seeded.disableToolCalls();
        assertEquals(
                wire,
                ReplyParser.parse(seeded, tokenizer.encode(wire)).text(),
                "prompt-owned ambiguous whitespace is not response text");

        ReplyParser partial = template.parser(tokenizer);
        partial.disableToolCalls();
        assertEquals(
                "{\"name\":",
                ReplyParser.parse(partial, tokenizer.encode("{\"name\":")).text(),
                "partial call-shaped JSON is not lost");

        ReplyParser lifecycle = template.parser(tokenizer);
        lifecycle.disableToolCalls();
        lifecycle.disableToolCalls();
        lifecycle.feed(tokenizer.encode("{").intAt(0));
        assertThrows(IllegalStateException.class, lifecycle::disableToolCalls);
        lifecycle.finish();
        assertThrows(IllegalStateException.class, lifecycle::disableToolCalls);

        // the model's own merge of the same wire, which tokenizer.encode does not reproduce
        Message generatedTokenization =
                ReplyParser.parse(
                        template.parser(tokenizer),
                        IntSequence.of(
                                        new int[] {
                                            5018, 609, 794, 330, 456, 70464, 498, 330, 14105, 794,
                                                    5324,
                                            9103, 794, 330, 53954, 16417, 498, 330, 3928, 794, 330,
                                                    66,
                                            41347, 32075
                                        })
                                .concat(
                                        IntSequence.of(
                                                SpecialTokens.require(tokenizer, "<|eot_id|>"))));
        assertTrue(generatedTokenization.content().getFirst() instanceof Content.ToolCall);

        Message variant =
                ReplyParser.parse(
                        template.parser(tokenizer),
                        tokenizer.encode(
                                "{\"type\": \"function\", \"function\": \"get_weather\", "
                                        + "\"parameters\": {\"city\": \"Paris\"}}"));
        assertEquals("get_weather", ((Content.ToolCall) variant.content().getFirst()).name());

        Message ordinary =
                ReplyParser.parse(template.parser(tokenizer), tokenizer.encode("No tool needed."));
        assertEquals("No tool needed.", ordinary.text());
    }

    @Test
    void miniCpmCdataCallIsClaimedOnlyWhenToolsAreEnabled() throws Exception {
        Tokenizer tokenizer = tokenizer("hf.co/openbmb/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q8_0.gguf");
        MiniCpm5ChatTemplate template = new MiniCpm5ChatTemplate(tokenizer);
        String first = " name=\"search\"><param name=\"query\"><![CDATA[a < b & c\nnext]]>";
        String second = "<param name=\"limit\">3";
        IntSequence wire =
                IntSequence.of(SpecialTokens.require(tokenizer, "<function"))
                        .concat(tokenizer.encode(first))
                        .concat(IntSequence.of(SpecialTokens.require(tokenizer, "</param>")))
                        .concat(tokenizer.encode(second))
                        .concat(IntSequence.of(SpecialTokens.require(tokenizer, "</param>")))
                        .concat(IntSequence.of(SpecialTokens.require(tokenizer, "</function>")));

        Content.ToolCall call =
                (Content.ToolCall)
                        ReplyParser.parse(template.parser(tokenizer), wire).content().getFirst();
        assertEquals("search", call.name());
        assertEquals(Map.of("query", "a < b & c\nnext", "limit", "3"), call.arguments());

        ReplyParser disabled = template.parser(tokenizer);
        disabled.disableToolCalls();
        Message text = ReplyParser.parse(disabled, wire);
        assertEquals(first + "</param>" + second + "</param>", text.text());
        assertEquals(wire, ((Content.Text) text.content().getFirst()).verbatim());
        assertTrue(text.content().stream().noneMatch(Content.ToolCall.class::isInstance));
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
                MemoryAllocators.ofArena(Arena.ofAuto()), new float[tokenizer.vocabulary().size()]);
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
