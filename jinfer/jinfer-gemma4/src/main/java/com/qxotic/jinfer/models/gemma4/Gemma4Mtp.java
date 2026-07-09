package com.qxotic.jinfer.models.gemma4;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.kernels.*;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;

/**
 * Gemma 4 MTP (multi-token prediction / self-speculative draft head), arch {@code
 * gemma4-assistant}. A 4-layer draft transformer that, at a decoded position, predicts the NEXT
 * token from the backbone's final hidden and the just-sampled token, then chains its own hidden to
 * draft further.
 *
 * <p>Structure (from llama.cpp {@code src/models/gemma4-assistant.cpp}, verified against the
 * sidecar GGUF): {@code xh = concat(backbone.tokEmbd[next]*sqrt(1536), backboneHidden[1536])} →
 * {@code pre_projection[3072,256]} → 4 layers → {@code output_norm} → tied {@code
 * token_embd[256,vocab]} for draft logits; {@code post_projection[256,1536]} produces the next
 * backbone-dim hidden to chain depth&gt;1.
 *
 * <p>The draft layers project Q ONLY (no K/V weights, {@code shared_kv_layers=4}) and attend
 * against the BACKBONE's KV cache: SWA draft layers (0-2) read backbone layer {@code
 * ownKvLayers-2}, the full draft layer (3) reads {@code ownKvLayers-1} - exactly jinfer's {@link
 * Gemma4.Configuration#kvSourceLayer} mapping, and llama.cpp's assistant KV-share map. This
 * shared-KV Q-only attention is the crux the draft-forward (Stage 2) implements and the
 * token-identity gate verifies.
 *
 * <p>This class is the load + capability surface (Stage 1): a shape-verified sidecar reader. The
 * draft forward and the speculative loop are the subsequent stages; {@link Gemma4} exposes MTP
 * through the core {@code MultiToken} seam once they land.
 */
public final class Gemma4Mtp {

    /** Draft-transformer geometry, read from the {@code gemma4-assistant.*} sidecar metadata. */
    public record Config(
            int embeddingLength, // draft dim (256)
            int backboneDim, // embedding_length_out (1536) - the backbone hidden the draft consumes
            int numberOfLayers, // 4
            int feedForwardLength, // 2048
            int numberOfHeads, // 4
            int numberOfKvHeads, // 1 (unused for projection - drafts share backbone KV)
            int headSizeFull, // 512
            int headSizeSWA, // 256
            int slidingWindow, // 512
            float rmsNormEps,
            float ropeThetaFull, // 1e6
            float ropeThetaSWA, // 1e4
            boolean[] isSWA, // [true,true,true,false]
            int vocabularySize) {

        int headSize(int layer) {
            return isSWA[layer] ? headSizeSWA : headSizeFull;
        }

        int queryDim(int layer) {
            return numberOfHeads * headSize(layer);
        }
    }

    /**
     * Draft weights. {@code tokenEmbeddings} is tied: the draft input embedding AND the draft LM
     * head.
     */
    public static final class Weights {
        final FloatTensor tokenEmbeddings; // token_embd.weight [256, vocab], tied head
        final FloatTensor preProjection; // nextn.pre_projection [3072, 256]
        final FloatTensor postProjection; // nextn.post_projection [256, 1536]
        final F32FloatTensor outputNorm; // output_norm.weight [256]
        final F32FloatTensor[] attnNorm, attnQNorm, attnPostNorm, ffnNorm, ffnPostNorm;
        final FloatTensor[] wq, wo, ffnGate, ffnUp, ffnDown;
        final float[] layerOutputScales;
        final float[] ropeFreqFactors; // rope_freqs.weight (full-layer rope factors)

        Weights(
                FloatTensor tokenEmbeddings,
                FloatTensor preProjection,
                FloatTensor postProjection,
                F32FloatTensor outputNorm,
                F32FloatTensor[] attnNorm,
                F32FloatTensor[] attnQNorm,
                F32FloatTensor[] attnPostNorm,
                F32FloatTensor[] ffnNorm,
                F32FloatTensor[] ffnPostNorm,
                FloatTensor[] wq,
                FloatTensor[] wo,
                FloatTensor[] ffnGate,
                FloatTensor[] ffnUp,
                FloatTensor[] ffnDown,
                float[] layerOutputScales,
                float[] ropeFreqFactors) {
            this.tokenEmbeddings = tokenEmbeddings;
            this.preProjection = preProjection;
            this.postProjection = postProjection;
            this.outputNorm = outputNorm;
            this.attnNorm = attnNorm;
            this.attnQNorm = attnQNorm;
            this.attnPostNorm = attnPostNorm;
            this.ffnNorm = ffnNorm;
            this.ffnPostNorm = ffnPostNorm;
            this.wq = wq;
            this.wo = wo;
            this.ffnGate = ffnGate;
            this.ffnUp = ffnUp;
            this.ffnDown = ffnDown;
            this.layerOutputScales = layerOutputScales;
            this.ropeFreqFactors = ropeFreqFactors;
        }
    }

    private final Config config;
    private final Weights weights;

    private Gemma4Mtp(Config config, Weights weights) {
        this.config = config;
        this.weights = weights;
    }

    public Config config() {
        return config;
    }

    public Weights weights() {
        return weights;
    }

    /**
     * Loads and shape-verifies the MTP sidecar. {@code backboneVocab} is the backbone's vocab,
     * which the tied draft head must match (the draft predicts backbone tokens).
     */
    public static Gemma4Mtp loadSidecar(Path sidecar, int backboneVocab) throws IOException {
        try (FileChannel fc = FileChannel.open(sidecar, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(fc, sidecar.toString());
            String arch = gguf.getValueOrDefault(String.class, "general.architecture", "");
            if (!arch.equals("gemma4-assistant")) {
                throw new IllegalArgumentException(
                        "not a gemma4-assistant MTP sidecar: arch=" + arch);
            }
            Config config = readConfig(gguf, backboneVocab);
            Map<String, GGMLTensorEntry> tensors = ModelLoader.loadTensors(fc, gguf);
            Weights weights = loadWeights(tensors, config);
            return new Gemma4Mtp(config, weights);
        }
    }

    private static Config readConfig(GGUF gguf, int backboneVocab) {
        String p = "gemma4-assistant.";
        int layers = gguf.getValue(int.class, p + "block_count");
        int headFull = gguf.getValue(int.class, p + "attention.key_length");
        int headSWA = gguf.getValue(int.class, p + "attention.key_length_swa");
        // isSWA per layer: full-attention layers carry the larger head (key_length); SWA carry
        // key_length_swa.
        // The GGUF flags the pattern via the per-layer attn_q width, matched here to the Gemma
        // convention
        // [SWA, SWA, SWA, FULL] (blk0-2 head=key_length_swa, blk3 head=key_length).
        boolean[] isSWA = new boolean[layers];
        for (int i = 0; i < layers; i++) isSWA[i] = i < layers - 1;
        return new Config(
                gguf.getValue(int.class, p + "embedding_length"),
                gguf.getValue(int.class, p + "embedding_length_out"),
                layers,
                gguf.getValue(int.class, p + "feed_forward_length"),
                gguf.getValue(int.class, p + "attention.head_count"),
                gguf.getValue(int.class, p + "attention.head_count_kv"),
                headFull,
                headSWA,
                gguf.getValue(int.class, p + "attention.sliding_window"),
                gguf.getValueOrDefault(float.class, p + "attention.layer_norm_rms_epsilon", 1e-6f),
                gguf.getValueOrDefault(float.class, p + "rope.freq_base", 1000000f),
                gguf.getValueOrDefault(float.class, p + "rope.freq_base_swa", 10000f),
                isSWA,
                backboneVocab);
    }

    private static Weights loadWeights(Map<String, GGMLTensorEntry> t, Config c) {
        int n = c.numberOfLayers();
        int dim = c.embeddingLength();

        FloatTensor tokenEmbeddings = req(t, "token_embd.weight", c.vocabularySize() * (long) dim);
        FloatTensor preProjection =
                req(t, "nextn.pre_projection.weight", 2L * c.backboneDim() * dim);
        FloatTensor postProjection =
                req(t, "nextn.post_projection.weight", (long) dim * c.backboneDim());
        F32FloatTensor outputNorm = f32(t, "output_norm.weight", dim);

        F32FloatTensor[] attnNorm = new F32FloatTensor[n], attnQNorm = new F32FloatTensor[n];
        F32FloatTensor[] attnPostNorm = new F32FloatTensor[n],
                ffnNorm = new F32FloatTensor[n],
                ffnPostNorm = new F32FloatTensor[n];
        FloatTensor[] wq = new FloatTensor[n], wo = new FloatTensor[n];
        FloatTensor[] ffnGate = new FloatTensor[n],
                ffnUp = new FloatTensor[n],
                ffnDown = new FloatTensor[n];
        float[] outScales = new float[n];

        for (int i = 0; i < n; i++) {
            String b = "blk." + i + ".";
            int qDim = c.queryDim(i);
            int hs = c.headSize(i);
            attnNorm[i] = f32(t, b + "attn_norm.weight", dim);
            wq[i] = req(t, b + "attn_q.weight", (long) dim * qDim);
            wo[i] = req(t, b + "attn_output.weight", (long) qDim * dim);
            attnQNorm[i] = f32(t, b + "attn_q_norm.weight", hs);
            attnPostNorm[i] = f32(t, b + "post_attention_norm.weight", dim);
            ffnNorm[i] = f32(t, b + "ffn_norm.weight", dim);
            ffnGate[i] = req(t, b + "ffn_gate.weight", (long) dim * c.feedForwardLength());
            ffnUp[i] = req(t, b + "ffn_up.weight", (long) dim * c.feedForwardLength());
            ffnDown[i] = req(t, b + "ffn_down.weight", (long) c.feedForwardLength() * dim);
            ffnPostNorm[i] = f32(t, b + "post_ffw_norm.weight", dim);
            GGMLTensorEntry scale = t.get(b + "layer_output_scale.weight");
            if (scale == null)
                throw new IllegalStateException("missing " + b + "layer_output_scale.weight");
            outScales[i] = ModelLoader.loadQuantized(scale).getFloat(0);
        }
        float[] ropeFreqFactors = ModelLoader.ropeFreqFactors(t);
        return new Weights(
                tokenEmbeddings,
                preProjection,
                postProjection,
                outputNorm,
                attnNorm,
                attnQNorm,
                attnPostNorm,
                ffnNorm,
                ffnPostNorm,
                wq,
                wo,
                ffnGate,
                ffnUp,
                ffnDown,
                outScales,
                ropeFreqFactors);
    }

    private static FloatTensor req(
            Map<String, GGMLTensorEntry> t, String name, long expectedElems) {
        GGMLTensorEntry e = t.get(name);
        if (e == null) throw new IllegalStateException("MTP sidecar missing tensor " + name);
        long got = 1;
        for (int d : e.shape()) got *= d;
        if (got != expectedElems) {
            throw new IllegalStateException(
                    name
                            + " shape "
                            + java.util.Arrays.toString(e.shape())
                            + " = "
                            + got
                            + " elems, expected "
                            + expectedElems);
        }
        return ModelLoader.loadQuantized(e);
    }

    private static F32FloatTensor f32(
            Map<String, GGMLTensorEntry> t, String name, long expectedElems) {
        return (F32FloatTensor) req(t, name, expectedElems);
    }
}
