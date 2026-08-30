package com.qxotic.jinfer.models.qwen35;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

final class Qwen35MtpLoadTest {

    private static final String MODEL_REF = "hf.co/unsloth/Qwen3.5-9B-MTP-GGUF:Q4_0";

    @Test
    void loadsDenseAndMoeMtpThroughTheOrdinaryLayerArrays() {
        try (Arena arena = Arena.ofConfined()) {
            for (boolean moe : new boolean[] {false, true}) {
                Qwen35.Configuration config = config(moe);
                Qwen35.Weights weights = Qwen35.loadWeights(tensors(config, arena), config);

                assertEquals(3, weights.attnNorm().length);
                assertNotNull(weights.attnQ()[config.mtpLayer()]);
                assertNotNull(weights.nextn());
                assertSame(weights.tokenEmbedding(), weights.nextn().tokenEmbedding());
                assertSame(weights.outputNorm(), weights.nextn().outputNorm());
                assertSame(weights.outputWeight(), weights.nextn().outputWeight());
                if (moe) assertNotNull(weights.moeRouter()[config.mtpLayer()]);
                else assertNotNull(weights.ffnGate()[config.mtpLayer()]);
            }
        }
    }

    @Test
    void rejectsMissingMtpTensorsAtLoadTime() {
        try (Arena arena = Arena.ofConfined()) {
            Qwen35.Configuration config = config(false);
            for (String missing :
                    new String[] {"blk.2.nextn.eh_proj.weight", "blk.2.attn_q.weight"}) {
                Map<String, MemoryView<MemorySegment>> tensors = tensors(config, arena);
                tensors.remove(missing);
                IllegalArgumentException error =
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> Qwen35.loadWeights(tensors, config));
                assertTrue(error.getMessage().contains(missing));
            }
        }
    }

    @Test
    void metadataRatherThanTensorGuessingControlsMtp() {
        try (Arena arena = Arena.ofConfined()) {
            Qwen35.Configuration config = withoutMtp();
            Map<String, MemoryView<MemorySegment>> tensors = tensors(config, arena);
            tensors.put("blk.1.nextn.eh_proj.weight", tensors.get("token_embd.weight"));

            IllegalArgumentException error =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> Qwen35.loadWeights(tensors, config));
            assertTrue(error.getMessage().contains("nextn_predict_layers=0"));
        }
    }

    @Test
    @Tag("integration")
    void realEmbeddedMtpModelLoads() throws Exception {
        Path path = TestModels.require(MODEL_REF);
        Qwen35 model = Qwen35.loadModel(path, Arena.ofAuto());
        assertEquals(1, model.configuration().nextnPredictLayers());
        assertEquals(
                model.configuration().numberOfLayers() + 1, model.configuration().storedLayers());
        assertTrue(model.configuration().isFullAttention()[model.configuration().mtpLayer()]);
        assertTrue(model.speculationReady());
    }

    static Map<String, MemoryView<MemorySegment>> tensors(
            Qwen35.Configuration config, Arena arena) {
        MemoryArena<MemorySegment> memory = MemoryAllocators.ofArena(arena);
        int dim = config.embeddingLength();
        Map<String, MemoryView<MemorySegment>> tensors = new HashMap<>();
        tensors.put("token_embd.weight", f32(memory, config.vocabularySize(), dim));
        tensors.put("output_norm.weight", f32(memory, dim));
        tensors.put("output.weight", f32(memory, config.vocabularySize(), dim));
        for (int layer = 0; layer < config.storedLayers(); layer++) {
            String p = "blk." + layer + ".";
            tensors.put(p + "attn_norm.weight", f32(memory, dim));
            tensors.put(p + "post_attention_norm.weight", f32(memory, dim));
            if (config.isFullAttention()[layer]) {
                tensors.put(p + "attn_q.weight", f32(memory, 2L * config.queryDim(), dim));
                tensors.put(p + "attn_k.weight", f32(memory, config.kvDim(), dim));
                tensors.put(p + "attn_v.weight", f32(memory, config.kvDim(), dim));
                tensors.put(p + "attn_output.weight", f32(memory, dim, config.queryDim()));
                tensors.put(p + "attn_q_norm.weight", f32(memory, config.headSize()));
                tensors.put(p + "attn_k_norm.weight", f32(memory, config.headSize()));
            } else {
                tensors.put(p + "attn_qkv.weight", f32(memory, config.convChannels(), dim));
                tensors.put(p + "attn_gate.weight", f32(memory, config.ssmInnerSize(), dim));
                tensors.put(p + "ssm_alpha.weight", f32(memory, config.ssmTimeStepRank(), dim));
                tensors.put(p + "ssm_beta.weight", f32(memory, config.ssmTimeStepRank(), dim));
                tensors.put(p + "ssm_out.weight", f32(memory, dim, config.ssmInnerSize()));
                tensors.put(
                        p + "ssm_conv1d.weight",
                        f32(memory, config.convChannels(), config.ssmConvKernel()));
                tensors.put(p + "ssm_a", f32(memory, config.ssmTimeStepRank()));
                tensors.put(p + "ssm_dt.bias", f32(memory, config.ssmTimeStepRank()));
                tensors.put(p + "ssm_norm.weight", f32(memory, config.headVDim()));
            }
            if (config.isMoE()) {
                tensors.put(p + "ffn_gate_inp.weight", f32(memory, config.expertCount(), dim));
                tensors.put(
                        p + "ffn_gate_exps.weight",
                        expertStack(
                                memory,
                                config.expertCount(),
                                config.expertFeedForwardLength(),
                                config.embeddingLength()));
                tensors.put(
                        p + "ffn_up_exps.weight",
                        expertStack(
                                memory,
                                config.expertCount(),
                                config.expertFeedForwardLength(),
                                config.embeddingLength()));
                tensors.put(
                        p + "ffn_down_exps.weight",
                        expertStack(
                                memory,
                                config.expertCount(),
                                config.embeddingLength(),
                                config.expertFeedForwardLength()));
                tensors.put(p + "ffn_gate_inp_shexp.weight", f32(memory, dim));
                tensors.put(
                        p + "ffn_gate_shexp.weight",
                        f32(memory, config.expertSharedFeedForwardLength(), dim));
                tensors.put(
                        p + "ffn_up_shexp.weight",
                        f32(memory, config.expertSharedFeedForwardLength(), dim));
                tensors.put(
                        p + "ffn_down_shexp.weight",
                        f32(memory, dim, config.expertSharedFeedForwardLength()));
            } else {
                tensors.put(p + "ffn_gate.weight", f32(memory, config.hiddenDim(), dim));
                tensors.put(p + "ffn_up.weight", f32(memory, config.hiddenDim(), dim));
                tensors.put(p + "ffn_down.weight", f32(memory, dim, config.hiddenDim()));
            }
        }
        if (config.hasMtp()) {
            String nextn = "blk." + config.mtpLayer() + ".nextn.";
            tensors.put(nextn + "enorm.weight", f32(memory, dim));
            tensors.put(nextn + "hnorm.weight", f32(memory, dim));
            tensors.put(nextn + "eh_proj.weight", f32(memory, dim, 2L * dim));
        }
        return tensors;
    }

    private static MemoryView<MemorySegment> f32(MemoryArena<MemorySegment> memory, long... shape) {
        return Views.allocateF32(memory, shape);
    }

    private static MemoryView<MemorySegment> expertStack(
            MemoryArena<MemorySegment> memory, int experts, int rows, int cols) {
        return Views.allocateF32(memory, experts, rows, cols);
    }

    static Qwen35.Configuration withoutMtp() {
        Qwen35.Configuration mtp = config(false);
        return new Qwen35.Configuration(
                mtp.embeddingLength(),
                mtp.numberOfLayers(),
                0,
                mtp.numberOfHeads(),
                mtp.numberOfKeyValueHeads(),
                mtp.headSize(),
                mtp.vocabularySize(),
                mtp.contextLength(),
                mtp.rmsNormEps(),
                mtp.ropeTheta(),
                mtp.ropeDimensionCount(),
                mtp.ropeDimensionSections(),
                mtp.hiddenDim(),
                new boolean[] {true, false},
                mtp.ssmInnerSize(),
                mtp.ssmGroupCount(),
                mtp.ssmTimeStepRank(),
                mtp.ssmStateSize(),
                mtp.ssmConvKernel(),
                0,
                0,
                0,
                0);
    }

    static Qwen35.Configuration config(boolean moe) {
        return new Qwen35.Configuration(
                4,
                2,
                1,
                2,
                1,
                2,
                8,
                16,
                1e-5f,
                10_000f,
                2,
                new int[] {1, 0, 0},
                moe ? 0 : 8,
                new boolean[] {true, false, true},
                4,
                1,
                2,
                2,
                3,
                moe ? 2 : 0,
                moe ? 1 : 0,
                moe ? 8 : 0,
                moe ? 4 : 0);
    }
}
