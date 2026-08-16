package com.qxotic.jinfer.langchain4j;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.chat.ReplyLanguage;
import com.qxotic.jinfer.chat.ToolCallSyntax;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Measures the constrained-decode machinery, tokenizer-only: no weights, no forward passes. */
@Tag("integration")
class GrammarCostProbe {

    static final Map<String, Object> SCHEMA =
            Map.of(
                    "type", "object",
                    "properties",
                            Map.of(
                                    "city", Map.of("type", "string"),
                                    "temperature_c", Map.of("type", "number")),
                    "required", List.of("city", "temperature_c"));

    @Test
    void probe() throws Exception {
        Path model =
                TestModels.require("hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf");
        GGUF g;
        try (FileChannel ch = FileChannel.open(model, StandardOpenOption.READ)) {
            g = ModelLoader.readGguf(ch, model.toString());
        }
        Tokenizer tok = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(g);
        int vocab = tok.vocabulary().size();
        System.out.println("vocab=" + vocab);
        float[] arr = new float[vocab];
        MemoryView<MemorySegment> logits =
                Views.wrap(MemorySegment.ofArray(arr), DataType.FP32, Shape.flat(vocab));

        // 1. schema -> compiled spec (cold, then the (source, vocab) cache hit)
        long t0 = System.nanoTime();
        Grammar.Spec spec = Grammar.fromSchema(SCHEMA, tok);
        long compileCold = System.nanoTime() - t0;
        t0 = System.nanoTime();
        Grammar.fromSchema(SCHEMA, tok);
        long compileWarm = System.nanoTime() - t0;
        System.out.printf(
                "schema compile: cold %.2f ms, cached %.3f ms%n",
                compileCold / 1e6, compileWarm / 1e6);

        // 2. the constrained walk: cold states compute full-vocab masks, warm states hit the
        //    per-state cache; mask application (bitmask -> -inf sweep) is paid every token
        IntSequence doc = tok.encode("{\"city\": \"Munich\", \"temperature_c\": 18.5}");
        System.out.println("doc tokens=" + doc.length());
        t0 = System.nanoTime();
        Grammar.Cursor cold = spec.cursor();
        for (int i = 0; i < doc.length(); i++) {
            Arrays.fill(arr, 0f);
            cold.maskLogits(logits);
            cold.tryAdvance(doc.intAt(i));
        }
        long coldWalk = System.nanoTime() - t0;
        int rounds = 200;
        t0 = System.nanoTime();
        for (int r = 0; r < rounds; r++) {
            Grammar.Cursor c = spec.cursor();
            for (int i = 0; i < doc.length(); i++) {
                Arrays.fill(arr, 0f);
                c.maskLogits(logits);
                c.tryAdvance(doc.intAt(i));
            }
        }
        long warmWalk = System.nanoTime() - t0;
        t0 = System.nanoTime();
        for (int r = 0; r < rounds; r++) {
            for (int i = 0; i < doc.length(); i++) {
                Arrays.fill(arr, 0f);
            }
        }
        long fillOnly = System.nanoTime() - t0;
        System.out.printf(
                "constrained walk: cold %.2f ms total (%.2f ms/token, per-state masks computed"
                        + " once), warm %.1f us/token of which %.1f us is the bench's own logits"
                        + " refill -> mask apply ~%.1f us/token%n",
                coldWalk / 1e6,
                coldWalk / 1e6 / doc.length(),
                warmWalk / 1e3 / (rounds * doc.length()),
                fillOnly / 1e3 / (rounds * doc.length()),
                (warmWalk - fillOnly) / 1e3 / (rounds * doc.length()));

        // 3. the per-REQUIRED-request cost: a full tools+schema Selection (compile + validate +
        //    closures + forcedPrefix), like the prototype's constrainedAuto
        t0 = System.nanoTime();
        ReplyLanguage.Selection sel =
                ReplyLanguage.Selection.of(
                        ReplyLanguage.spans(
                                "<think>",
                                "</think>",
                                "<|tool_call_start|>",
                                "<|tool_call_end|>",
                                ToolCallSyntax::parseBlock,
                                ReplyLanguage.mark("<|im_end|>"),
                                ReplyLanguage.gbnf(Grammar.schemaGbnf(SCHEMA))),
                        tok);
        long selBuild = System.nanoTime() - t0;
        System.out.printf("tools+schema Selection.of: %.2f ms%n", selBuild / 1e6);

        // 4. walk feed on FREE content (the AUTO/think path): pure parse, no mask
        ReplyLanguage.Walk walk = sel.walk();
        IntSequence think = tok.encode("Let me check the weather for Munich now.");
        int open = SpecialTokens.require(tok, "<think>");
        walk.feed(open);
        t0 = System.nanoTime();
        int reps = 2000;
        for (int r = 0; r < reps; r++) {
            for (int i = 0; i < think.length(); i++) walk.feed(think.intAt(i));
        }
        long feedTime = System.nanoTime() - t0;
        System.out.printf(
                "walk.feed on free text: %.2f us/token%n",
                feedTime / 1e3 / (double) (reps * think.length()));
    }
}
