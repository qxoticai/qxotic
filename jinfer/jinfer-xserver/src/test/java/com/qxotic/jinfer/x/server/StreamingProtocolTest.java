package com.qxotic.jinfer.x.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;

import com.qxotic.jinfer.x.llm.Generator;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.OptionalInt;
import org.junit.jupiter.api.Test;

class StreamingProtocolTest {

    @Test
    void usageIsEmittedOnlyInTheRequestedUsageChunk() {
        Map<String, Object> finish = new LinkedHashMap<>(Map.of("choices", List.of("finish")));
        Map<String, Object> usageOnly = new LinkedHashMap<>();
        List<Map<String, Object>> chunks =
                Server.streamEndChunks(
                        Map.of("stream_options", Map.of("include_usage", true)),
                        reply(),
                        finish,
                        usageOnly);

        assertEquals(2, chunks.size());
        assertNull(chunks.getFirst().get("usage"));
        assertEquals(List.of(), chunks.getLast().get("choices"));
        assertEquals(
                7,
                Values.intValue(
                        Values.asObject(chunks.getLast().get("usage"), "usage").get("total_tokens"),
                        -1));
    }

    @Test
    void usageChunkIsAbsentUnlessExplicitlyRequested() {
        for (Map<String, Object> request :
                List.<Map<String, Object>>of(
                        Map.of(), Map.of("stream_options", Map.of("include_usage", false)))) {
            Map<String, Object> finish = new LinkedHashMap<>();
            List<Map<String, Object>> chunks =
                    Server.streamEndChunks(request, reply(), finish, new LinkedHashMap<>());
            assertEquals(1, chunks.size());
            assertFalse(finish.containsKey("usage"));
        }
    }

    private static Reply reply() {
        var result =
                new Generator.GenerationResult(
                        new int[] {1, 2, 3},
                        OptionalInt.empty(),
                        Generator.FinishReason.LENGTH,
                        Duration.ofMillis(1),
                        Duration.ofMillis(2));
        return new Reply(result, 4, 1, "ok", null, List.of(), "length", null);
    }
}
