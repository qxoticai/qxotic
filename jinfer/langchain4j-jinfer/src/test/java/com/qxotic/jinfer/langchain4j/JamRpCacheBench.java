package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Files;
import java.util.Arrays;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Prices jam's repacked-weight cache on the model it exclusively serves (MoE expert gemms): decode
 * throughput on LFM2.5-8B-A1B, cache rung on vs off. Run twice against the knob build:
 *
 * <pre>
 * JAVA_TOOL_OPTIONS=-Djam.native.library.path=.../libjam.so.1.0.0 \
 *   [JAM_RP_CACHE=off] mvn -f langchain4j-jinfer/pom.xml test \
 *   -Dsurefire.excludedGroups= -Dgroups=bench -Dtest=JamRpCacheBench
 * </pre>
 *
 * If off is within noise of on, the cache does not earn its bytes (up to the full expert set
 * duplicated per distinct-address load) and the simplest fix is deleting the rung.
 */
@Tag("bench")
class JamRpCacheBench {

    @Test
    void moeDecodeThroughput() {
        java.nio.file.Path model =
                System.getProperty("jinfer.benchModel") != null
                        ? java.nio.file.Path.of(System.getProperty("jinfer.benchModel"))
                        : ModelFixture.LFM25_8B_Q8.path();
        Assumptions.assumeTrue(Files.exists(model));
        System.out.println(
                "JAM_RP_CACHE="
                        + String.valueOf(System.getenv("JAM_RP_CACHE"))
                        + "  lib="
                        + System.getProperty("jam.native.library.path", "bundled"));
        try (JinferChatModel m =
                JinferChatModel.builder()
                        .modelPath(model)
                        .contextLength(4096)
                        .maxOutputTokens(160)
                        .build()) {
            chat(m); // warmup: JIT + (cache-on) repack fill
            double[] tokPerS = new double[3];
            for (int i = 0; i < tokPerS.length; i++) {
                long t0 = System.nanoTime();
                ChatResponse r = chat(m);
                double s = (System.nanoTime() - t0) / 1e9;
                int out = r.tokenUsage().outputTokenCount();
                tokPerS[i] = out / s;
                System.out.printf(
                        "rep %d: %d tokens in %.2fs = %.2f tok/s%n", i, out, s, tokPerS[i]);
            }
            Arrays.sort(tokPerS);
            System.out.printf("MEDIAN %.2f tok/s%n", tokPerS[1]);
        }
    }

    private static ChatResponse chat(JinferChatModel m) {
        return m.chat(
                UserMessage.from(
                        "Write a paragraph about rivers. Keep going until you are cut off."));
    }
}
