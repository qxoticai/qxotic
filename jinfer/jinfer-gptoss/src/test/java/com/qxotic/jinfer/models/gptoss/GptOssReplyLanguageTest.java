package com.qxotic.jinfer.models.gptoss;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.TokenRuns;
import com.qxotic.jinfer.chat.Tool;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.llm.Tokenizers;
import com.qxotic.jinfer.testkit.ModelFixture;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * The gpt-oss migration gates, tokenizer-only. DIFFERENTIAL: the reply-language walk (now {@code
 * parser()}) must agree with {@link HarmonyReplyParser} - kept as the reference - over canonical
 * Harmony streams, verbatim ids included. WIRE LAW, strengthened: the forced-call selection must
 * ADMIT the family's own rendered call, arguments included, at every token - the law that makes
 * schema-bound forcing safe to ship.
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class GptOssReplyLanguageTest {

    Tokenizer tokenizer;
    GptOssTurnTemplate template;

    @BeforeAll
    void load() throws Exception {
        Path model = ModelFixture.GPTOSS_20B_Q8.path();
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model);
        GGUF gguf;
        try (FileChannel ch = FileChannel.open(model, StandardOpenOption.READ)) {
            gguf = ModelLoader.readGguf(ch, model.toString());
        }
        tokenizer = Tokenizers.fromGGUF(gguf);
        template = new GptOssTurnTemplate(tokenizer, "2026-07-25");
    }

    TokenRuns runs() {
        return new TokenRuns(tokenizer);
    }

    int id(String spelling) {
        return SpecialTokens.require(tokenizer, spelling);
    }

    /** Both parsers over one stream; returns the walk's message after asserting agreement. */
    Message agree(IntSequence stream) {
        Message walk = ReplyParser.parse(template.parser(), stream);
        Message reference = ReplyParser.parse(new HarmonyReplyParser(tokenizer), stream);
        assertEquals(reference.toString(), walk.toString(), "walk and reference must agree");
        return walk;
    }

    @Test
    void analysisThenFinalAgreesVerbatimIncluded() {
        IntSequence stream =
                runs().id(id("<|channel|>"))
                        .text("analysis")
                        .id(id("<|message|>"))
                        .text("The user greets; respond warmly.")
                        .id(id("<|end|>"))
                        .id(id("<|start|>"))
                        .text("assistant")
                        .id(id("<|channel|>"))
                        .text("final")
                        .id(id("<|message|>"))
                        .text("Hello! Comment ça va?")
                        .id(id("<|return|>"))
                        .build();
        Message m = agree(stream);
        Part.Reasoning r = (Part.Reasoning) m.content().get(0);
        assertEquals("The user greets; respond warmly.", r.text());
        assertTrue(r.verbatim() != null, "the analysis-retention echo needs the verbatim ids");
        assertEquals("Hello! Comment ça va?", m.text());
    }

    @Test
    void aCanonicalCallAgreesWithArgsOnlyVerbatim() {
        IntSequence stream =
                runs().id(id("<|channel|>"))
                        .text("analysis")
                        .id(id("<|message|>"))
                        .text("Need the weather.")
                        .id(id("<|end|>"))
                        .id(id("<|channel|>"))
                        .text("commentary to=functions.get_weather ")
                        .id(id("<|constrain|>"))
                        .text("json")
                        .id(id("<|message|>"))
                        .text("{\"city\": \"Zürich\"}")
                        .id(id("<|call|>"))
                        .build();
        Message m = agree(stream);
        Part.ToolCall c =
                (Part.ToolCall)
                        m.content().stream()
                                .filter(p -> p instanceof Part.ToolCall)
                                .findFirst()
                                .orElseThrow();
        assertEquals("get_weather", c.name());
        assertEquals(Map.of("city", "Zürich"), c.arguments());
        assertEquals(
                tokenizer.encode("{\"city\": \"Zürich\"}").toList(),
                c.verbatim().toList(),
                "verbatim is the ARGS BODY only - what the echo splices");
    }

    @Test
    void aMalformedCallDropsWithoutEndingTheReply() {
        IntSequence stream =
                runs().id(id("<|channel|>"))
                        .text("commentary to=functions.get_weather ")
                        .id(id("<|constrain|>"))
                        .text("json")
                        .id(id("<|message|>"))
                        .text("{\"city\": ")
                        .id(id("<|end|>"))
                        .id(id("<|channel|>"))
                        .text("final")
                        .id(id("<|message|>"))
                        .text("I could not call the tool.")
                        .id(id("<|return|>"))
                        .build();
        Message m = agree(stream);
        assertTrue(
                m.content().stream().noneMatch(p -> p instanceof Part.ToolCall),
                "a payload that never held a parseable object is no call");
        assertEquals("I could not call the tool.", m.text(), "the reply CONTINUES past the drop");
    }

    @Test
    void aCommentaryPreambleStreamsAsContent() {
        IntSequence stream =
                runs().id(id("<|channel|>"))
                        .text("commentary")
                        .id(id("<|message|>"))
                        .text("Fetching the forecast now.")
                        .id(id("<|end|>"))
                        .id(id("<|channel|>"))
                        .text("final")
                        .id(id("<|message|>"))
                        .text("Done.")
                        .id(id("<|return|>"))
                        .build();
        Message m = agree(stream);
        assertEquals("Fetching the forecast now.Done.", m.text());
    }

    // ---- the strengthened wire law: the forced selection admits its own rendered call ----

    static final Tool WEATHER =
            new Tool(
                    "get_weather",
                    "{\"name\":\"get_weather\",\"parameters\":{\"type\":\"object\","
                            + "\"properties\":{\"city\":{\"type\":\"string\"}},"
                            + "\"required\":[\"city\"]}}");
    static final Tool REFRESH =
            new Tool("refresh_cache", "{\"name\":\"refresh_cache\",\"parameters\":{}}");

    @Test
    void theForcedSelectionAdmitsItsOwnRenderedCallEveryToken() {
        ReplyLanguage.Selection sel = template.forcedCall(List.of(WEATHER)).orElseThrow();
        int[] prefix = sel.forcedPrefix();
        assertTrue(prefix.length > 0 && prefix[0] == id("<|channel|>"));

        ReplyLanguage.Walk walk = sel.walk();
        for (int t : prefix) walk.feed(t);
        IntSequence args = tokenizer.encode("{\"city\": \"Paris\"}");
        F32FloatTensor logits =
                F32FloatTensor.allocate(Arena.ofAuto(), tokenizer.vocabulary().size());
        int[] argTokens = args.toArray();
        for (int t : argTokens) {
            for (int i = 0; i < tokenizer.vocabulary().size(); i++) logits.setFloat(i, 0f);
            assertTrue(walk.maskLogits(logits), "the walk must stay live through its own wire");
            assertTrue(
                    logits.getFloat(t) == 0f,
                    "the schema grammar must admit the canonical args token " + t);
            walk.feed(t);
        }
        assertTrue(walk.accepted(), "a balanced schema payload is an acceptable end");
        walk.feed(id("<|call|>"));
        assertTrue(walk.accepted() && !walk.ended());
        Message m = walk.finish();
        Part.ToolCall c = (Part.ToolCall) m.content().get(0);
        assertEquals("get_weather", c.name());
        assertEquals(Map.of("city", "Paris"), c.arguments());
    }

    @Test
    void aMultiToolForcedSelectionBranchesOnlyIntoOfferedNames() {
        // two tools behind the shared header: candidacy walks both, the mask is their union -
        // an UNOFFERED name's diverging token is unsamplable at the branch point
        ReplyLanguage.Selection sel = template.forcedCall(List.of(WEATHER, REFRESH)).orElseThrow();
        ReplyLanguage.Walk walk = sel.walk();
        walk.feed(id("<|channel|>"));
        for (int t : tokenizer.encode("commentary to=functions.").toArray()) walk.feed(t);
        F32FloatTensor logits =
                F32FloatTensor.allocate(Arena.ofAuto(), tokenizer.vocabulary().size());
        for (int i = 0; i < tokenizer.vocabulary().size(); i++) logits.setFloat(i, 0f);
        assertTrue(walk.maskLogits(logits));
        int get = tokenizer.encode("get_weather").toArray()[0];
        int refresh = tokenizer.encode("refresh_cache").toArray()[0];
        int delete = tokenizer.encode("delete_everything").toArray()[0];
        assertTrue(logits.getFloat(get) == 0f, "an offered name's first token is admissible");
        assertTrue(logits.getFloat(refresh) == 0f, "both offered names are admissible");
        assertTrue(
                logits.getFloat(delete) == Float.NEGATIVE_INFINITY,
                "an unoffered name is unsamplable at the branch");
        // committing to the SECOND tool works end to end
        walk.feed(refresh);
        for (int t : tokenizer.encode("_cache ").toArray()) walk.feed(t);
        walk.feed(id("<|constrain|>"));
        for (int t : tokenizer.encode("json").toArray()) walk.feed(t);
        walk.feed(id("<|message|>"));
        for (int t : tokenizer.encode("{}").toArray()) walk.feed(t);
        assertTrue(walk.accepted());
        assertEquals("refresh_cache", ((Part.ToolCall) walk.finish().content().get(0)).name());
    }

    @Test
    void anInventedArgumentKeyIsUnsamplable() {
        // the toolbench_rapidapi_key class: the schema admits only declared property names
        ReplyLanguage.Selection sel = template.forcedCall(List.of(WEATHER)).orElseThrow();
        ReplyLanguage.Walk walk = sel.walk();
        for (int t : sel.forcedPrefix()) walk.feed(t);
        for (int t : tokenizer.encode("{\"").toArray()) walk.feed(t);
        F32FloatTensor logits =
                F32FloatTensor.allocate(Arena.ofAuto(), tokenizer.vocabulary().size());
        for (int i = 0; i < tokenizer.vocabulary().size(); i++) logits.setFloat(i, 0f);
        assertTrue(walk.maskLogits(logits));
        int city = tokenizer.encode("city").toArray()[0];
        int invented = tokenizer.encode("toolbench").toArray()[0];
        assertTrue(logits.getFloat(city) == 0f, "the declared key is admissible");
        assertTrue(
                logits.getFloat(invented) == Float.NEGATIVE_INFINITY,
                "an invented key's first token is unsamplable");
    }

    @Test
    void anAutoCallParsesWhateverNameTheModelWrote() {
        // AUTO is deliberately name-free: the parser reports, the framework judges (the walk
        // never saw the offered set - the language is tools-independent)
        IntSequence stream =
                runs().id(id("<|channel|>"))
                        .text("commentary to=functions.totally_unoffered ")
                        .id(id("<|constrain|>"))
                        .text("json")
                        .id(id("<|message|>"))
                        .text("{}")
                        .id(id("<|call|>"))
                        .build();
        Message m = agree(stream);
        assertEquals("totally_unoffered", ((Part.ToolCall) m.content().get(0)).name());
    }

    @Test
    void theNoSpaceConstrainVariantStillParses() {
        // the old parser defended "get_time<|constrain|>json" explicitly - the space is optional
        IntSequence stream =
                runs().id(id("<|channel|>"))
                        .text("commentary to=functions.get_time")
                        .id(id("<|constrain|>"))
                        .text("json")
                        .id(id("<|message|>"))
                        .text("{}")
                        .id(id("<|call|>"))
                        .build();
        Message m = agree(stream);
        assertEquals("get_time", ((Part.ToolCall) m.content().get(0)).name());
    }

    @Test
    void aNonFunctionsRecipientDropsWithoutEndingTheReply() {
        // browser.search is legal Harmony: the region completes, the parser filters it out, and
        // the reply CONTINUES (it used to truncate here; the body itself stays silent - the one
        // narrowed delta vs the old parser, which showed it as content)
        IntSequence stream =
                runs().id(id("<|channel|>"))
                        .text("commentary to=browser.search ")
                        .id(id("<|constrain|>"))
                        .text("json")
                        .id(id("<|message|>"))
                        .text("{\"q\": \"weather\"}")
                        .id(id("<|end|>"))
                        .id(id("<|channel|>"))
                        .text("final")
                        .id(id("<|message|>"))
                        .text("Here is the weather.")
                        .id(id("<|return|>"))
                        .build();
        Message m = ReplyParser.parse(template.parser(), stream);
        assertTrue(m.content().stream().noneMatch(p -> p instanceof Part.ToolCall));
        assertEquals("Here is the weather.", m.text(), "the reply continues past the drop");
    }

    @Test
    void aConstrainAdornedFinalHeaderStreamsAsContent() {
        // the JSON-response-format shape: <|channel|>final <|constrain|>json<|message|>{...}
        IntSequence stream =
                runs().id(id("<|channel|>"))
                        .text("final ")
                        .id(id("<|constrain|>"))
                        .text("json")
                        .id(id("<|message|>"))
                        .text("{\"answer\": 42}")
                        .id(id("<|return|>"))
                        .build();
        Message m = agree(stream);
        assertEquals("{\"answer\": 42}", m.text());
    }

    @Test
    void aForcedNoParameterToolCannotBeDecorated() {
        ReplyLanguage.Selection sel = template.forcedCall(List.of(REFRESH)).orElseThrow();
        ReplyLanguage.Walk walk = sel.walk();
        for (int t : sel.forcedPrefix()) walk.feed(t);
        // the empty-schema grammar admits only an empty object: the noParameterTool skip-gate
        // in the E2E battery becomes a hard impossibility here
        F32FloatTensor logits =
                F32FloatTensor.allocate(Arena.ofAuto(), tokenizer.vocabulary().size());
        for (int t : tokenizer.encode("{}").toArray()) {
            for (int i = 0; i < tokenizer.vocabulary().size(); i++) logits.setFloat(i, 0f);
            assertTrue(walk.maskLogits(logits));
            assertTrue(logits.getFloat(t) == 0f, "the empty object is admissible");
            walk.feed(t);
        }
        assertTrue(walk.accepted());
    }
}
