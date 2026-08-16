package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.boundary.Arenas;
import com.qxotic.jinfer.x.chat.Models;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import java.lang.foreign.Arena;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * ENFORCEMENT: use-after-free of a borrowed weights arena must be a teaching {@link
 * IllegalStateException} - never a SIGSEGV - for EVERY model family. Each family's forward pass
 * must call the safety canary before its first raw weight read; a port that forgets (or grows a
 * second gather without one - Qwen3's segmented path taught that lesson) fails here by crashing the
 * fork VM, which is exactly the regression this battery exists to catch.
 *
 * <p>Cheap by construction: loading is mmap + tokenizer only, no generation ever succeeds - the
 * arena is freed before the first forward. Families whose GGUF is absent are skipped and counted.
 */
@Tag("integration")
class WeightsCanaryIT {

    static final List<String> CHAT_FAMILIES =
            List.of(
                    "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf", // lfm2
                    "hf.co/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf", // qwen35
                    "hf.co/unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf", // gemma4
                    "hf.co/unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q8_0.gguf", // gptoss (harmony)
                    "hf.co/unsloth/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf", // llama
                    "hf.co/unsloth/Ministral-3-3B-Instruct-2512-GGUF/Ministral-3-3B-Instruct-2512-Q8_0.gguf", // mistral (llama port)
                    "hf.co/openbmb/MiniCPM5-1B-GGUF/MiniCPM5-1B-Q8_0.gguf", // minicpm (llama port)
                    "hf.co/ibm-granite/granite-4.1-3b-GGUF/granite-4.1-3b-Q8_0.gguf", // granite
                    "hf.co/ggml-org/SmolLM3-3B-GGUF/SmolLM3-Q4_K_M.gguf", // smollm3 (llama port)
                    "hf.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF/nvidia_Nemotron-Cascade-2-30B-A3B-Q8_0.gguf", // nemotron_h
                    "hf.co/prism-ml/Bonsai-27B-gguf/Bonsai-27B-Q1_0.gguf"); // qwen35moe (Q1_0)

    @Test
    void everyChatFamilyFailsFastOnFreedWeights() throws Exception {
        int covered = 0;
        for (String family : CHAT_FAMILIES) {
            var familyPath = TestModels.find(family);
            if (familyPath.isEmpty()) continue;
            Arena arena = Arenas.newCrossThread();
            JinferChatModel borrowed;
            try {
                var loaded = Models.load(familyPath.get(), arena);
                borrowed = JinferChatModel.builder().model(loaded).maxOutputTokens(8).build();
            } catch (Throwable t) {
                arena.close();
                throw new AssertionError(family + ": failed to load", t);
            }
            try {
                arena.close(); // the owner frees the weights before the first request
                IllegalStateException e =
                        assertThrows(
                                IllegalStateException.class,
                                () -> borrowed.chat(UserMessage.from("hi")),
                                family);
                assertTrue(e.getMessage().contains("freed"), family + ": " + e.getMessage());
            } finally {
                borrowed.close();
            }
            covered++;
        }
        Assumptions.assumeTrue(covered > 0, "no chat family GGUF on disk");
        System.out.println("weights-canary battery: " + covered + " chat families enforced");
    }

    /**
     * The STATE's guarantee, isolated: on a borrowed instance, {@code close()} frees only the state
     * arena - the weights stay alive, so the weights canary passes and the ONLY guard against raw
     * reads of freed state buffers is the state's own closed flag. One test per surface (the flag
     * is shared machinery, family-independent), weights kept alive throughout.
     */
    @Test
    void closedPipelineWithLiveWeightsFailsLoudly() throws Exception {
        try (Arena arena = Arenas.newCrossThread()) {
            // chat: the engine's lifecycle gate
            String chatFixture = "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf";
            var chatPath = TestModels.find(chatFixture);
            Assumptions.assumeTrue(chatPath.isPresent(), "no chat GGUF on disk");
            JinferChatModel chat =
                    JinferChatModel.builder()
                            .model(Models.load(chatPath.get(), arena))
                            .maxOutputTokens(8)
                            .build();
            chat.close();
            assertThrows(
                    IllegalStateException.class, () -> chat.chat(UserMessage.from("hi")), "chat");

            // embedding: the state's closed flag
            String embedFixture =
                    "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf";
            var embedPath = TestModels.find(embedFixture);
            if (embedPath.isPresent()) {
                JinferEmbeddingModel embed =
                        JinferEmbeddingModel.builder()
                                .model(
                                        com.qxotic.jinfer.x.chat.Models.loadEmbedder(
                                                embedPath.get(), arena))
                                .contextLength(256)
                                .build();
                embed.close();
                assertThrows(IllegalStateException.class, () -> embed.embed("hi"), "embed");
            }

            // scoring: same flag, reranker surface
            String rerankFixture =
                    "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf";
            var rerankPath = TestModels.find(rerankFixture);
            if (rerankPath.isPresent()) {
                JinferScoringModel score =
                        JinferScoringModel.builder()
                                .model(
                                        com.qxotic.jinfer.x.chat.Models.loadReranker(
                                                rerankPath.get(), arena))
                                .contextLength(512)
                                .build();
                score.close();
                assertThrows(
                        IllegalStateException.class,
                        () -> score.scoreAll(List.of(TextSegment.from("d")), "q"),
                        "score");
            }
        }
    }

    /**
     * The dilemma recurses at the CORE layer: {@code newState(ctx, batch, arena)} puts the state's
     * buffers in a CALLER-owned arena - closing it under a live state is the state-side twin of the
     * weights mistake, and no adapter flag guards it. {@code RuntimeState.exclusively()} is the
     * choke point (every forward enters the state first, every family), so the canary lives there
     * once: freed state arena, live weights, next pass must be a teaching ISE.
     */
    @Test
    void stateOverACallerArenaFailsFastWhenFreed() throws Exception {
        String fixture = "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf";
        var fixturePath = TestModels.require(fixture);
        try (Arena weights = Arenas.newCrossThread()) {
            var loaded = com.qxotic.jinfer.x.chat.Models.loadEmbedder(fixturePath, weights);
            Arena stateArena = Arenas.newCrossThread();
            var state = loaded.model().newState(512, 64, new PanamaMemoryArena(stateArena));
            stateArena.close(); // the caller frees the state's buffers under the state
            IllegalStateException e =
                    assertThrows(
                            IllegalStateException.class,
                            () -> loaded.embedAll(state, 512, List.of("hi"), v -> {}));
            assertTrue(e.getMessage().contains("freed"), e.getMessage());
        }
    }

    @Test
    void everyEmbedderFamilyFailsFastOnFreedWeights() throws Exception {
        int covered = 0;
        for (String family :
                List.of(
                        "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf",
                        "hf.co/LiquidAI/LFM2.5-Embedding-350M-GGUF/LFM2.5-Embedding-350M-Q8_0.gguf")) {
            var familyPath = TestModels.find(family);
            if (familyPath.isEmpty()) continue;
            Arena arena = Arenas.newCrossThread();
            JinferEmbeddingModel borrowed;
            try {
                var loaded = com.qxotic.jinfer.x.chat.Models.loadEmbedder(familyPath.get(), arena);
                borrowed = JinferEmbeddingModel.builder().model(loaded).contextLength(256).build();
            } catch (Throwable t) {
                arena.close();
                throw new AssertionError(family + ": failed to load", t);
            }
            try {
                arena.close();
                IllegalStateException e =
                        assertThrows(
                                IllegalStateException.class, () -> borrowed.embed("hello"), family);
                assertTrue(e.getMessage().contains("freed"), family + ": " + e.getMessage());
            } finally {
                borrowed.close();
            }
            covered++;
        }
        Assumptions.assumeTrue(covered > 0, "no embedder GGUF on disk");
    }

    @Test
    void everyRerankerFamilyFailsFastOnFreedWeights() throws Exception {
        int covered = 0;
        for (String family :
                List.of(
                        "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf",
                        "hf.co/LiquidAI/LFM2.5-ColBERT-350M-GGUF/LFM2.5-ColBERT-350M-Q8_0.gguf")) {
            var familyPath = TestModels.find(family);
            if (familyPath.isEmpty()) continue;
            Arena arena = Arenas.newCrossThread();
            JinferScoringModel borrowed;
            try {
                var loaded = com.qxotic.jinfer.x.chat.Models.loadReranker(familyPath.get(), arena);
                borrowed = JinferScoringModel.builder().model(loaded).contextLength(512).build();
            } catch (Throwable t) {
                arena.close();
                throw new AssertionError(family + ": failed to load", t);
            }
            try {
                arena.close();
                IllegalStateException e =
                        assertThrows(
                                IllegalStateException.class,
                                () ->
                                        borrowed.scoreAll(
                                                List.of(TextSegment.from("a document")), "a query"),
                                family);
                assertTrue(e.getMessage().contains("freed"), family + ": " + e.getMessage());
            } finally {
                borrowed.close();
            }
            covered++;
        }
        Assumptions.assumeTrue(covered > 0, "no reranker GGUF on disk");
    }
}
