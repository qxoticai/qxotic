package com.qxotic.jinfer.x.models.qwen35;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.ContentKey;
import com.qxotic.jinfer.x.cache.PromptCache;
import com.qxotic.jinfer.x.chat.ChatEngine;
import com.qxotic.jinfer.x.chat.LoadedModel;
import com.qxotic.jinfer.x.llm.Generator;
import com.qxotic.jinfer.x.llm.Sampler;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.file.Path;
import java.time.Duration;
import java.util.AbstractMap;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The loaded MTP state is cacheable regardless of which decode loop a request selects. */
final class Qwen35MtpCacheLifecycleTest {

    private static final Generator.Constraints THREE_TOKENS =
            new Generator.Constraints(3, Duration.ZERO, Set.of());
    private static final ContentKey SEED = ContentKey.sha256(new byte[] {3, 5});

    @Test
    void engineDepthSelectsTheLoopWithoutChangingCachedState() {
        try (Arena weights = Arena.ofShared()) {
            Qwen35 model = tinyModel(weights);
            var loaded =
                    new LoadedModel<>(
                            model,
                            TinyTokenizer.INSTANCE,
                            "",
                            Set.of(),
                            SEED,
                            Optional.empty(),
                            LoadedModel.SamplingDefaults.NONE);
            var options = new PromptCache.Options(1, 16, 0, null, false);
            try (ChatEngine engine = new ChatEngine(loaded, "tiny-qwen35-mtp", options)) {
                assertTrue(engine.speculationReady(), "weights are loaded independently of depth");
                int[] firstPrompt = {1, 2, 3};

                engine.speculationDepth(0);
                ChatEngine.Outcome plain =
                        engine.generate(
                                List.of(Batch.prefill(firstPrompt)),
                                Sampler.ARGMAX,
                                3,
                                Duration.ZERO,
                                token -> true);
                assertTrue(plain.speculated().isEmpty());

                int[] secondPrompt = extend(firstPrompt, plain.result().tokens(), 4);
                engine.speculationDepth(2);
                ChatEngine.Outcome mtp =
                        engine.generate(
                                List.of(Batch.prefill(secondPrompt)),
                                Sampler.ARGMAX,
                                3,
                                Duration.ZERO,
                                token -> true);
                assertEquals(PromptCache.Tier.SESSION, mtp.tier());
                assertTrue(mtp.speculated().isPresent());

                engine.speculationDepth(0);
                ChatEngine.Outcome plainAgain =
                        engine.generate(
                                List.of(
                                        Batch.prefill(
                                                extend(secondPrompt, mtp.result().tokens(), 5))),
                                Sampler.ARGMAX,
                                3,
                                Duration.ZERO,
                                token -> true);
                assertEquals(PromptCache.Tier.SESSION, plainAgain.tier());
                assertTrue(plainAgain.speculated().isEmpty());
            }
        }
    }

    @Test
    void depthIsHarmlessWhenTheModelHasNoMtpLayer() {
        try (Arena weights = Arena.ofShared()) {
            Qwen35 model = tinyModel(weights, false);
            var loaded =
                    new LoadedModel<>(
                            model,
                            TinyTokenizer.INSTANCE,
                            "",
                            Set.of(),
                            SEED,
                            Optional.empty(),
                            LoadedModel.SamplingDefaults.NONE);
            var options = new PromptCache.Options(1, 16, 0, null, false);
            try (ChatEngine engine = new ChatEngine(loaded, "tiny-qwen35", options)) {
                assertFalse(engine.speculationReady());
                engine.speculationDepth(2);

                int[] prompt = {1, 2, 3};
                ChatEngine.Outcome first =
                        engine.generate(
                                List.of(Batch.prefill(prompt)),
                                Sampler.ARGMAX,
                                3,
                                Duration.ZERO,
                                token -> true);
                assertTrue(first.speculated().isEmpty());

                ChatEngine.Outcome second =
                        engine.generate(
                                List.of(Batch.prefill(extend(prompt, first.result().tokens(), 4))),
                                Sampler.ARGMAX,
                                3,
                                Duration.ZERO,
                                token -> true);
                assertEquals(PromptCache.Tier.SESSION, second.tier());
                assertTrue(second.speculated().isEmpty());
            }
        }
    }

    @Test
    void retainedStateSwitchesFromPlainToMtpAndBack() {
        try (Arena weights = Arena.ofShared()) {
            Qwen35 model = tinyModel(weights);
            assertTrue(model.speculationReady(), "MTP is loaded even while plain decode is chosen");

            try (PromptCache<Qwen35.State> cache = cache(model, 1, 0)) {
                int[] prompt = {1, 2, 3};
                Pass plain = run(cache, model, List.of(Batch.prefill(prompt)), false);
                assertFalse(plain.speculated());

                Pass mtp =
                        run(
                                cache,
                                model,
                                List.of(Batch.prefill(extend(prompt, plain.committed(), 4))),
                                true);
                Pass replay = freshRun(model, extend(prompt, plain.committed(), 4), true);
                assertEquals(PromptCache.Tier.SESSION, mtp.tier());
                assertArrayEquals(replay.committed(), mtp.committed());
                assertTrue(mtp.speculated());
                assertTrue(mtp.drafted() > 0);
                assertTrue(mtp.accepted() > 0, "the retained MTP carry produces accepted drafts");
                assertEquals(0, mtp.auditViolations());
            }

            try (PromptCache<Qwen35.State> cache = cache(model, 1, 0)) {
                int[] prompt = {1, 2, 3};
                Pass mtp = run(cache, model, List.of(Batch.prefill(prompt)), true);
                Pass plain =
                        run(
                                cache,
                                model,
                                List.of(Batch.prefill(extend(prompt, mtp.committed(), 4))),
                                false);
                Pass replay = freshRun(model, extend(prompt, mtp.committed(), 4), false);
                assertEquals(PromptCache.Tier.SESSION, plain.tier());
                assertArrayEquals(replay.committed(), plain.committed());
                assertFalse(plain.speculated());
            }
        }
    }

    @Test
    void definedMtpPrefixServesBothDecodeModes() {
        try (Arena weights = Arena.ofShared()) {
            Qwen35 model = tinyModel(weights);
            try (PromptCache<Qwen35.State> cache = cache(model, 0, 1 << 20)) {
                List<Batch> prompt = List.of(Batch.prefill(new int[] {1, 2}), Batch.step(3));
                cache.define(prompt);

                Pass plain = run(cache, model, prompt, false);
                Pass mtp = run(cache, model, prompt, true);
                assertEquals(PromptCache.Tier.BLOCKS, plain.tier());
                assertEquals(PromptCache.Tier.BLOCKS, mtp.tier());
                assertArrayEquals(
                        freshRun(model, new int[] {1, 2, 3}, false).committed(), plain.committed());
                assertArrayEquals(
                        freshRun(model, new int[] {1, 2, 3}, true).committed(), mtp.committed());
                assertTrue(mtp.accepted() > 0, "the restored MTP carry produces accepted drafts");
                assertEquals(0, mtp.auditViolations());
            }
        }
    }

    @Test
    void persistedMtpPrefixServesPlainAndMtpAfterReopen(@TempDir Path directory) {
        try (Arena weights = Arena.ofShared()) {
            Qwen35 model = tinyModel(weights);
            Path catalog = directory.resolve("qwen35-mtp.jkvf");
            List<Batch> prompt = List.of(Batch.prefill(new int[] {1, 2}), Batch.step(3));
            var options = new PromptCache.Options(0, 16, 1 << 20, catalog, false);
            try (PromptCache<Qwen35.State> writer = PromptCache.of(model, SEED, options)) {
                writer.define(prompt);
                writer.save();
            }

            try (PromptCache<Qwen35.State> reader =
                    PromptCache.of(model, SEED, options.withCatalog(catalog, true))) {
                Pass plain = run(reader, model, prompt, false);
                Pass mtp = run(reader, model, prompt, true);
                assertEquals(PromptCache.Tier.BLOCKS, plain.tier());
                assertEquals(PromptCache.Tier.BLOCKS, mtp.tier());
                assertArrayEquals(
                        freshRun(model, new int[] {1, 2, 3}, false).committed(), plain.committed());
                assertArrayEquals(
                        freshRun(model, new int[] {1, 2, 3}, true).committed(), mtp.committed());
                assertTrue(mtp.accepted() > 0, "persisted MTP carry produces accepted drafts");
                assertEquals(0, mtp.auditViolations());
            }
        }
    }

    private static Pass run(
            PromptCache<Qwen35.State> cache, Qwen35 model, List<Batch> prompt, boolean speculate) {
        return cache.serve(
                prompt,
                (state, serving) -> {
                    if (speculate) {
                        int[] violations = {0};
                        var result =
                                model.speculate(
                                        state,
                                        Sampler.ARGMAX,
                                        THREE_TOKENS,
                                        2,
                                        null,
                                        (token, target) -> {
                                            if (token != target) violations[0]++;
                                        });
                        int[] committed = result.committed().toArray();
                        serving.adopt(committed);
                        return new Pass(
                                serving.tier(),
                                committed,
                                true,
                                result.drafted(),
                                result.accepted(),
                                violations[0]);
                    }
                    var committed = new java.util.ArrayList<Integer>();
                    Generator.generate(
                            model,
                            state,
                            List.of(),
                            Sampler.ARGMAX,
                            THREE_TOKENS,
                            new Generator.GenerationListener() {
                                @Override
                                public boolean onToken(int token) {
                                    return true;
                                }

                                @Override
                                public void onIngested(int token) {
                                    serving.tail(token);
                                    committed.add(token);
                                }
                            });
                    return new Pass(
                            serving.tier(),
                            committed.stream().mapToInt(Integer::intValue).toArray(),
                            false,
                            0,
                            0,
                            0);
                });
    }

    private static PromptCache<Qwen35.State> cache(Qwen35 model, int retained, long bytes) {
        return PromptCache.of(
                model, SEED, new PromptCache.Options(retained, 16, bytes, null, false));
    }

    private static Pass freshRun(Qwen35 model, int[] prompt, boolean speculate) {
        try (PromptCache<Qwen35.State> fresh = cache(model, 0, 0)) {
            return run(fresh, model, List.of(Batch.prefill(prompt)), speculate);
        }
    }

    private static int[] extend(int[] prompt, int[] committed, int token) {
        int[] out = Arrays.copyOf(prompt, prompt.length + committed.length + 1);
        System.arraycopy(committed, 0, out, prompt.length, committed.length);
        out[out.length - 1] = token;
        return out;
    }

    private record Pass(
            PromptCache.Tier tier,
            int[] committed,
            boolean speculated,
            int drafted,
            int accepted,
            int auditViolations) {}

    private static Qwen35 tinyModel(Arena arena) {
        return tinyModel(arena, true);
    }

    private static Qwen35 tinyModel(Arena arena, boolean mtp) {
        Qwen35.Configuration c =
                mtp ? Qwen35MtpLoadTest.config(false) : Qwen35MtpLoadTest.withoutMtp();
        MemoryAllocator<MemorySegment> memory = new PanamaMemoryArena(arena);
        Map<String, MemoryView<MemorySegment>> t = new HashMap<>();
        float[] embedding = new float[c.vocabularySize() * c.embeddingLength()];
        for (int i = 0; i < embedding.length; i++)
            embedding[i] = (i % c.embeddingLength() + 1) / 8f;
        t.put("token_embd.weight", Views.fromFloatArray(memory, embedding));
        t.put("output_norm.weight", ones(memory, c.embeddingLength()));
        float[] head = new float[c.vocabularySize() * c.embeddingLength()];
        head[c.embeddingLength()] = 1f; // positive hidden -> token 1; zero carry -> token 0
        t.put("output.weight", Views.fromFloatArray(memory, head));

        for (int layer = 0; layer < c.storedLayers(); layer++) {
            String p = "blk." + layer + ".";
            t.put(p + "attn_norm.weight", ones(memory, c.embeddingLength()));
            t.put(p + "post_attention_norm.weight", ones(memory, c.embeddingLength()));
            if (c.isFullAttention()[layer]) {
                t.put(p + "attn_q.weight", zero(memory, 2L * c.queryDim() * c.embeddingLength()));
                t.put(p + "attn_k.weight", zero(memory, (long) c.kvDim() * c.embeddingLength()));
                t.put(p + "attn_v.weight", zero(memory, (long) c.kvDim() * c.embeddingLength()));
                t.put(
                        p + "attn_output.weight",
                        zero(memory, (long) c.embeddingLength() * c.queryDim()));
                t.put(p + "attn_q_norm.weight", ones(memory, c.headSize()));
                t.put(p + "attn_k_norm.weight", ones(memory, c.headSize()));
            } else {
                int channels = c.convChannels();
                t.put(p + "attn_qkv.weight", zero(memory, (long) channels * c.embeddingLength()));
                t.put(
                        p + "attn_gate.weight",
                        zero(memory, (long) c.ssmInnerSize() * c.embeddingLength()));
                t.put(
                        p + "ssm_alpha.weight",
                        zero(memory, (long) c.ssmTimeStepRank() * c.embeddingLength()));
                t.put(
                        p + "ssm_beta.weight",
                        zero(memory, (long) c.ssmTimeStepRank() * c.embeddingLength()));
                t.put(
                        p + "ssm_out.weight",
                        zero(memory, (long) c.embeddingLength() * c.ssmInnerSize()));
                t.put(p + "ssm_conv1d.weight", zero(memory, (long) channels * c.ssmConvKernel()));
                t.put(p + "ssm_a", zero(memory, c.ssmTimeStepRank()));
                t.put(p + "ssm_dt.bias", zero(memory, c.ssmTimeStepRank()));
                t.put(p + "ssm_norm.weight", ones(memory, c.headVDim()));
            }
            t.put(p + "ffn_gate.weight", zero(memory, (long) c.hiddenDim() * c.embeddingLength()));
            t.put(p + "ffn_up.weight", zero(memory, (long) c.hiddenDim() * c.embeddingLength()));
            t.put(p + "ffn_down.weight", zero(memory, (long) c.embeddingLength() * c.hiddenDim()));
        }
        if (mtp) {
            String nextn = "blk." + c.mtpLayer() + ".nextn.";
            t.put(nextn + "enorm.weight", ones(memory, c.embeddingLength()));
            t.put(nextn + "hnorm.weight", ones(memory, c.embeddingLength()));
            float[] projection = new float[2 * c.embeddingLength() * c.embeddingLength()];
            for (int d = 0; d < c.embeddingLength(); d++)
                projection[d * 2 * c.embeddingLength() + c.embeddingLength() + d] = 1f;
            t.put(nextn + "eh_proj.weight", Views.fromFloatArray(memory, projection));
        }
        return new Qwen35(c, null, Qwen35.loadWeights(t, c));
    }

    private static MemoryView<MemorySegment> zero(
            MemoryAllocator<MemorySegment> memory, long elements) {
        return Views.allocateF32(memory, elements);
    }

    private static MemoryView<MemorySegment> ones(
            MemoryAllocator<MemorySegment> memory, int elements) {
        float[] values = new float[elements];
        Arrays.fill(values, 1f);
        return Views.fromFloatArray(memory, values);
    }

    private enum TinyTokenizer implements Tokenizer {
        INSTANCE;

        private final Vocabulary vocabulary =
                new Vocabulary() {
                    @Override
                    public int size() {
                        return 8;
                    }

                    @Override
                    public String token(int id) {
                        return "t" + id;
                    }

                    @Override
                    public int id(String token) {
                        return Integer.parseInt(token.substring(1));
                    }

                    @Override
                    public boolean contains(int id) {
                        return id >= 0 && id < size();
                    }

                    @Override
                    public boolean contains(String token) {
                        return token.matches("t[0-7]");
                    }

                    @Override
                    public Iterator<Map.Entry<String, Integer>> iterator() {
                        return java.util.stream.IntStream.range(0, size())
                                .<Map.Entry<String, Integer>>mapToObj(
                                        i -> new AbstractMap.SimpleImmutableEntry<>(token(i), i))
                                .iterator();
                    }
                };

        @Override
        public Vocabulary vocabulary() {
            return vocabulary;
        }

        @Override
        public void encodeInto(CharSequence text, int from, int to, IntSequence.Builder out) {
            for (int i = from; i < to; i++) out.add(text.charAt(i) & 7);
        }

        @Override
        public int countTokens(CharSequence text, int from, int to) {
            return to - from;
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int from, ByteBuffer out) {
            if (from == tokens.length()) return 0;
            out.put((byte) ('0' + tokens.intAt(from)));
            return 1;
        }
    }
}
