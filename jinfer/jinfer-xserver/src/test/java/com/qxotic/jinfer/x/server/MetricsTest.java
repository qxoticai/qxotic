package com.qxotic.jinfer.x.server;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.x.llm.Generator;
import com.qxotic.jinfer.x.llm.SpeculativeDecoding;
import com.qxotic.toknroll.IntSequence;
import java.time.Duration;
import java.util.List;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;

class MetricsTest {

    @Test
    void exposesMtpAcceptanceWithoutOwningMtpPolicy() {
        var result =
                new Generator.GenerationResult(
                        new int[] {1, 2},
                        OptionalInt.empty(),
                        Generator.FinishReason.LENGTH,
                        Duration.ofMillis(4),
                        Duration.ofMillis(8));
        var speculation =
                new SpeculativeDecoding.SpeculationResult(
                        IntSequence.of(1, 2),
                        IntSequence.of(1, 2),
                        OptionalInt.empty(),
                        Generator.FinishReason.LENGTH,
                        Duration.ofMillis(8),
                        5,
                        3,
                        2);
        Metrics metrics = new Metrics();
        metrics.record(new Reply(result, 4, 1, "ok", null, List.of(), "length", speculation));
        try (Worker worker = new Worker(1)) {
            worker.start();
            String text = metrics.exposition(worker);
            assertTrue(text.contains("jinfer_speculation_drafted_tokens_total 5"), text);
            assertTrue(text.contains("jinfer_speculation_accepted_tokens_total 3"), text);
            assertTrue(text.contains("jinfer_speculation_forwards_total 2"), text);
        }
    }
}
