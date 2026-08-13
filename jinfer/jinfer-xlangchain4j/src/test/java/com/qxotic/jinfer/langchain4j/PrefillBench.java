package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Model-level prefill TTFT: a ~2k-token prompt, 1 output token, median of 3. The kernel-tier knob
 * is jam's own {@code JAM_ISA} env (e.g. {@code JAM_ISA=avx2} prices the pre-VNNI path); {@code
 * -Djinfer.benchModel} switches the GGUF (default LFM2.5-8B; pass a K-quant model to exercise the
 * K-quant bands).
 */
@Tag("bench")
class PrefillBench {

    @Test
    void prefillTtft() {
        Path model =
                System.getProperty("jinfer.benchModel") != null
                        ? Path.of(System.getProperty("jinfer.benchModel"))
                        : TestModels.find(
                                        "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf")
                                .orElse(
                                        Path.of(
                                                "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf"));
        Assumptions.assumeTrue(Files.exists(model));
        String para =
                "The river carves the valley and the valley steers the river; silt remembers"
                        + " every flood and the terraces record the argument between water and"
                        + " stone across ten thousand seasons of patient disagreement. ";
        String prompt = para.repeat(60) + "\nSummarize in one word.";
        System.out.println(
                "JAM_ISA="
                        + String.valueOf(System.getenv("JAM_ISA"))
                        + "  model="
                        + model.getFileName());
        try (JinferChatModel m =
                JinferChatModel.builder()
                        .modelPath(model)
                        .contextLength(4096)
                        .maxOutputTokens(1)
                        .build()) {
            m.chat(UserMessage.from(prompt)); // warmup
            double[] secs = new double[3];
            for (int i = 0; i < secs.length; i++) {
                long t0 = System.nanoTime();
                ChatResponse r = m.chat(UserMessage.from(prompt));
                secs[i] = (System.nanoTime() - t0) / 1e9;
                System.out.printf(
                        "rep %d: prefill %d tokens in %.2fs%n",
                        i, r.tokenUsage().inputTokenCount(), secs[i]);
            }
            Arrays.sort(secs);
            System.out.printf("PREFILL MEDIAN %.2fs%n", secs[1]);
        }
    }
}
