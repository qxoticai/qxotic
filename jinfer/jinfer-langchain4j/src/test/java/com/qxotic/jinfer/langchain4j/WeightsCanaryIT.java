package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.ModelFixture;
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

    static final List<ModelFixture.Gguf> CHAT_FAMILIES =
            List.of(
                    ModelFixture.LFM25_350M_Q8, // lfm2
                    ModelFixture.QWEN35_2B_Q8, // qwen35
                    ModelFixture.GEMMA4_E2B_Q8, // gemma4
                    ModelFixture.GPTOSS_20B_Q8, // gptoss (harmony)
                    ModelFixture.LLAMA32_1B_Q8, // llama
                    ModelFixture.MINISTRAL_3B_Q8, // mistral (llama port)
                    ModelFixture.MINICPM5_1B_Q8, // minicpm (llama port)
                    ModelFixture.GRANITE_41_3B_Q8, // granite
                    ModelFixture.SMOLLM3_Q4, // smollm3 (llama port)
                    ModelFixture.NEMOTRON_30B_Q8, // nemotron_h
                    ModelFixture.BONSAI_27B_Q1); // qwen35moe (Q1_0)

    @Test
    void everyChatFamilyFailsFastOnFreedWeights() throws Exception {
        int covered = 0;
        for (ModelFixture.Gguf family : CHAT_FAMILIES) {
            if (!family.present()) continue;
            Arena arena = Arena.ofShared();
            JinferChatModel borrowed;
            try {
                var loaded = Models.load(family.path(), arena);
                borrowed = JinferChatModel.builder().model(loaded).maxOutputTokens(8).build();
            } catch (Throwable t) {
                arena.close();
                throw new AssertionError(family.file() + ": failed to load", t);
            }
            try {
                arena.close(); // the owner frees the weights before the first request
                IllegalStateException e =
                        assertThrows(
                                IllegalStateException.class,
                                () -> borrowed.chat(UserMessage.from("hi")),
                                family.file());
                assertTrue(e.getMessage().contains("freed"), family.file() + ": " + e.getMessage());
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
        try (Arena arena = Arena.ofShared()) {
            // chat: the engine's lifecycle gate
            var chatFixture = ModelFixture.LFM25_350M_Q8;
            Assumptions.assumeTrue(chatFixture.present(), "no chat GGUF on disk");
            JinferChatModel chat =
                    JinferChatModel.builder()
                            .model(Models.load(chatFixture.path(), arena))
                            .maxOutputTokens(8)
                            .build();
            chat.close();
            assertThrows(
                    IllegalStateException.class, () -> chat.chat(UserMessage.from("hi")), "chat");

            // embedding: the state's closed flag
            var embedFixture = ModelFixture.QWEN3_EMBED_06B_Q8;
            if (embedFixture.present()) {
                JinferEmbeddingModel embed =
                        JinferEmbeddingModel.builder()
                                .model(Models.loadEmbedder(embedFixture.path(), arena))
                                .contextLength(256)
                                .build();
                embed.close();
                assertThrows(IllegalStateException.class, () -> embed.embed("hi"), "embed");
            }

            // scoring: same flag, reranker surface
            var rerankFixture = ModelFixture.QWEN3_RERANKER_06B_Q8;
            if (rerankFixture.present()) {
                JinferScoringModel score =
                        JinferScoringModel.builder()
                                .model(Models.loadReranker(rerankFixture.path(), arena))
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
     * weights mistake, and no adapter flag guards it. {@code BaseState.enter()} is the universal
     * choke point (every forward claims the state first, every family), so the canary lives there
     * once: freed state arena, live weights, next pass must be a teaching ISE.
     */
    @Test
    void stateOverACallerArenaFailsFastWhenFreed() throws Exception {
        var fixture = ModelFixture.QWEN3_EMBED_06B_Q8;
        Assumptions.assumeTrue(fixture.present(), "no embedder GGUF on disk");
        try (Arena weights = Arena.ofShared()) {
            var loaded = Models.loadEmbedder(fixture.path(), weights);
            Arena stateArena = Arena.ofShared();
            var state = loaded.model().newState(512, 64, stateArena);
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
        for (ModelFixture.Gguf family :
                List.of(ModelFixture.QWEN3_EMBED_06B_Q8, ModelFixture.LFM25_EMBEDDING_350M_Q8)) {
            if (!family.present()) continue;
            Arena arena = Arena.ofShared();
            JinferEmbeddingModel borrowed;
            try {
                var loaded = Models.loadEmbedder(family.path(), arena);
                borrowed = JinferEmbeddingModel.builder().model(loaded).contextLength(256).build();
            } catch (Throwable t) {
                arena.close();
                throw new AssertionError(family.file() + ": failed to load", t);
            }
            try {
                arena.close();
                IllegalStateException e =
                        assertThrows(
                                IllegalStateException.class,
                                () -> borrowed.embed("hello"),
                                family.file());
                assertTrue(e.getMessage().contains("freed"), family.file() + ": " + e.getMessage());
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
        for (ModelFixture.Gguf family :
                List.of(ModelFixture.QWEN3_RERANKER_06B_Q8, ModelFixture.LFM25_COLBERT_350M_Q8)) {
            if (!family.present()) continue;
            Arena arena = Arena.ofShared();
            JinferScoringModel borrowed;
            try {
                var loaded = Models.loadReranker(family.path(), arena);
                borrowed = JinferScoringModel.builder().model(loaded).contextLength(512).build();
            } catch (Throwable t) {
                arena.close();
                throw new AssertionError(family.file() + ": failed to load", t);
            }
            try {
                arena.close();
                IllegalStateException e =
                        assertThrows(
                                IllegalStateException.class,
                                () ->
                                        borrowed.scoreAll(
                                                List.of(TextSegment.from("a document")), "a query"),
                                family.file());
                assertTrue(e.getMessage().contains("freed"), family.file() + ": " + e.getMessage());
            } finally {
                borrowed.close();
            }
            covered++;
        }
        Assumptions.assumeTrue(covered > 0, "no reranker GGUF on disk");
    }
}
