package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.MediaEncodingCache;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.SpeculativeDecoding;
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
        metrics.recordPromptCache(PromptCache.Tier.SESSION, 5);
        metrics.recordPromptCache(PromptCache.Tier.BLOCKS, 3);
        metrics.recordPromptCache(PromptCache.Tier.FRESH, 0);
        for (Metrics.Outcome outcome : Metrics.Outcome.values()) metrics.record(outcome);
        try (Worker worker = new Worker(1)) {
            worker.start();
            String text =
                    metrics.exposition(
                            worker,
                            new PromptCache.Sample(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
                            new MediaEncodingCache.Sample(1, 2, 3, 4, 5, 6));
            assertTrue(text.contains("jinfer_generations_completed_total 1"), text);
            assertTrue(text.contains("jinfer_generation_requests_invalid_total 1"), text);
            assertTrue(text.contains("jinfer_generation_requests_rejected_total 1"), text);
            assertTrue(text.contains("jinfer_generation_requests_cancelled_total 1"), text);
            assertTrue(text.contains("jinfer_generation_requests_failed_total 1"), text);
            assertTrue(text.contains("jinfer_client_disconnects_total 1"), text);
            assertTrue(text.contains("jinfer_speculation_drafted_tokens_total 5"), text);
            assertTrue(text.contains("jinfer_speculation_accepted_tokens_total 3"), text);
            assertTrue(text.contains("jinfer_speculation_forwards_total 2"), text);
            assertTrue(
                    text.contains("jinfer_prompt_cache_requests_total{source=\"session\"} 1"),
                    text);
            assertTrue(text.contains("jinfer_prompt_cache_tokens_total{source=\"block\"} 3"), text);
            assertTrue(text.contains("jinfer_prompt_cache_session_count 1"), text);
            assertTrue(text.contains("jinfer_prompt_cache_session_limit 2"), text);
            assertTrue(text.contains("jinfer_prompt_cache_state_allocations_total 4"), text);
            assertTrue(text.contains("jinfer_prompt_cache_memory_usage_bytes 7"), text);
            assertTrue(
                    text.contains("jinfer_prompt_cache_block_lookups_total{result=\"hit\"} 9"),
                    text);
            assertTrue(
                    text.contains("jinfer_prompt_cache_block_lookups_total{result=\"miss\"} 10"),
                    text);
            assertTrue(
                    text.contains(
                            "jinfer_prompt_cache_block_removals_total{reason=\"evicted\"} 11"),
                    text);
            assertTrue(
                    text.contains(
                            "jinfer_prompt_cache_block_removals_total{reason=\"discarded\"} 12"),
                    text);
            assertTrue(text.contains("jinfer_prompt_cache_block_refusals_total 13"), text);
            assertTrue(text.contains("jinfer_media_cache_entry_count 1"), text);
            assertTrue(text.contains("jinfer_media_cache_memory_usage_bytes 2"), text);
            assertTrue(text.contains("jinfer_media_cache_memory_limit_bytes 3"), text);
            assertTrue(text.contains("jinfer_media_cache_lookups_total{result=\"hit\"} 4"), text);
            assertTrue(text.contains("jinfer_media_cache_lookups_total{result=\"miss\"} 5"), text);
            assertTrue(text.contains("jinfer_media_cache_refusals_total 6"), text);
            assertFalse(text.contains("jinfer_session_pool_hits_total"), text);
            assertFalse(text.contains("jinfer_cached_tokens_total"), text);
        }
    }
}
