package com.qxotic.jinfer.models.gemma4;

import static com.qxotic.jinfer.Norms.rmsnorm;

import com.qxotic.jinfer.Activations;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FlashAttention;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.Pair;
import com.qxotic.jinfer.RoPE;

/**
 * Gemma 4 MTP draft forward (Stage 2): one self-speculative draft step. Given the backbone's
 * pre-final-norm hidden for a position and the token sampled there, predicts the next-token
 * distribution through the 4-layer {@code gemma4-assistant} draft transformer.
 *
 * <p>The draft layers project Q ONLY and attend the BACKBONE's KV rings (shared-KV, {@code
 * shared_kv_layers=4}): the 3 SWA draft layers read backbone own-KV layer {@code ownKv-2}, the full
 * draft layer reads {@code ownKv-1} (llama.cpp {@code ctx_other} map). Geometry: draft 4 heads x
 * {256 SWA, 512 full}, backbone 1 KV head at the same head size -> GQA with {@code kvMul=4} over
 * the backbone's single-KV-head cache; {@link FlashAttention#flashDecode} with null batch buffers
 * reads exactly the cached window {@code [attStart, position]}.
 *
 * <p>Flow (from {@code gemma4-assistant.cpp}): {@code x = backbone.tokEmbd[token]*sqrt(1536)};
 * {@code cur = pre_proj @ concat(x, hidden)[3072] -> 256}; 4 layers (attn_norm, wq, q_norm, rope,
 * Q-only attn, wo, attn_post_norm, +res, ffn gelu-par, ffn_post_norm, +res, *out_scale); {@code
 * output_norm}; draft logits = tied {@code token_embd @ cur} (no softcap); {@code post_proj @ cur
 * -> 1536} chains depth&gt;1.
 */
public final class Gemma4MtpDecoder {

    private final Gemma4Mtp.Config cfg;
    private final Gemma4Mtp.Weights w;
    private final Gemma4 backbone;
    private final int backboneOwnKv;

    // rope tables per attention regime (same theta/head-size convention as the backbone)
    private final F32FloatTensor realSWA, imagSWA, realFull, imagFull;

    // scratch (single-token draft; small)
    private final F32FloatTensor xh, cur, xb, q, attn, hb, hb2, hNext;
    private final FloatTensor draftLogits;
    private final FlashAttention.DecodeScratch decodeScratch = new FlashAttention.DecodeScratch();

    public Gemma4MtpDecoder(Gemma4Mtp mtp, Gemma4 backbone) {
        this.cfg = mtp.config();
        this.w = mtp.weights();
        this.backbone = backbone;
        this.backboneOwnKv = backbone.config().ownKvLayers();
        int ctx = backbone.config().contextLength();

        Pair<float[], float[]> swa =
                RoPE.precomputeFreqsCis(ctx, cfg.headSizeSWA(), cfg.ropeThetaSWA());
        Pair<float[], float[]> full =
                w.ropeFreqFactors != null
                        ? RoPE.precomputeFreqsCisFromFreqs(
                                ctx, cfg.headSizeFull(), cfg.ropeThetaFull(), w.ropeFreqFactors)
                        : RoPE.precomputeFreqsCis(ctx, cfg.headSizeFull(), cfg.ropeThetaFull());
        this.realSWA = F32FloatTensor.of(swa.first());
        this.imagSWA = F32FloatTensor.of(swa.second());
        this.realFull = F32FloatTensor.of(full.first());
        this.imagFull = F32FloatTensor.of(full.second());

        int dim = cfg.embeddingLength();
        int maxQ = cfg.numberOfHeads() * cfg.headSizeFull();
        this.xh = F32FloatTensor.allocate(2 * cfg.backboneDim());
        this.cur = F32FloatTensor.allocate(dim);
        this.xb = F32FloatTensor.allocate(dim);
        this.q = F32FloatTensor.allocate(maxQ);
        this.attn = F32FloatTensor.allocate(maxQ);
        this.hb = F32FloatTensor.allocate(cfg.feedForwardLength());
        this.hb2 = F32FloatTensor.allocate(cfg.feedForwardLength());
        this.hNext = F32FloatTensor.allocate(cfg.backboneDim());
        this.draftLogits = F32FloatTensor.allocate(cfg.vocabularySize());
    }

    /**
     * Draft the next-token logits given {@code hidden} (the backbone pre-final-norm hidden of the
     * position where {@code token} was produced) at attention {@code position}. Returns the draft
     * logits (reused buffer); {@link #chainedHidden()} holds the 1536-dim hidden for depth&gt;1.
     */
    public FloatTensor draft(
            Gemma4.State backboneState,
            FloatTensor hidden,
            long hiddenOffset,
            int token,
            int position) {
        int bd = cfg.backboneDim(), dim = cfg.embeddingLength();
        float eps = cfg.rmsNormEps();

        // xh = concat( backbone.tokEmbd[token] * sqrt(bd) , hidden )
        FloatTensor btok = backbone.weights().tokenEmbeddings;
        float scale = (float) Math.sqrt(bd);
        for (int i = 0; i < bd; i++) xh.setFloat(i, btok.getFloat((long) token * bd + i) * scale);
        for (int i = 0; i < bd; i++) xh.setFloat(bd + i, hidden.getFloat(hiddenOffset + i));

        // cur = pre_projection @ xh   [3072] -> [256]
        w.preProjection.matmul(xh, cur, dim, 2 * bd);

        for (int l = 0; l < cfg.numberOfLayers(); l++) {
            boolean swa = cfg.isSWA()[l];
            int headSize = cfg.headSize(l), halfHead = headSize / 2, qDim = cfg.queryDim(l);
            F32FloatTensor real = swa ? realSWA : realFull, imag = swa ? imagSWA : imagFull;

            // attn: norm -> Q -> per-head q_norm + rope -> Q-only attention on backbone rings -> wo
            // -> post_norm -> +res
            rmsnorm(xb, 0, cur, 0, w.attnNorm[l], dim, eps);
            w.wq[l].matmul(xb, q, qDim, dim);
            for (int h = 0; h < cfg.numberOfHeads(); h++) {
                rmsnorm(q, h * headSize, q, h * headSize, w.attnQNorm[l], headSize, eps);
            }
            RoPE.applyNeox(q, 0, cfg.numberOfHeads(), headSize, halfHead, position, real, imag);

            int kvSrc = swa ? backboneOwnKv - 2 : backboneOwnKv - 1;
            int bkvDim = backbone.config().kvDim(kvSrc);
            int kvMul =
                    cfg.numberOfHeads() / backbone.config().numberOfKeyValueHeadsPerLayer()[kvSrc];
            int window = backbone.config().slidingWindow();
            int attStart = swa ? Math.max(0, position - window + 1) : 0;
            // draftState has no own KV: null batch buffers -> attend only backbone cache [attStart,
            // position]
            FlashAttention.flashDecode(
                    q,
                    attn,
                    backboneState.keyCache[kvSrc],
                    backboneState.valueCache[kvSrc],
                    null,
                    null,
                    cfg.numberOfHeads(),
                    position,
                    attStart,
                    headSize,
                    bkvDim,
                    kvMul,
                    1.0f,
                    swa ? window - 1 : 0,
                    null,
                    decodeScratch);
            w.wo[l].matmul(attn, xb, dim, qDim);
            rmsnorm(xb, 0, xb, 0, w.attnPostNorm[l], dim, eps);
            cur.addInPlace(0, xb, 0, dim);

            // ffn: norm -> gelu-par gate*up -> down -> post_norm -> +res, then * layer_output_scale
            rmsnorm(xb, 0, cur, 0, w.ffnNorm[l], dim, eps);
            w.ffnGate[l].matmul(xb, hb, cfg.feedForwardLength(), dim);
            w.ffnUp[l].matmul(xb, hb2, cfg.feedForwardLength(), dim);
            Activations.geluMultiply(hb, 0, hb2, 0, cfg.feedForwardLength());
            w.ffnDown[l].matmul(hb, xb, dim, cfg.feedForwardLength());
            rmsnorm(xb, 0, xb, 0, w.ffnPostNorm[l], dim, eps);
            cur.addInPlace(0, xb, 0, dim);
            final float outScale = w.layerOutputScales[l];
            cur.mapInPlace(0, dim, v -> v * outScale);
        }

        rmsnorm(cur, 0, cur, 0, w.outputNorm, dim, eps);
        w.tokenEmbeddings.matmul(
                cur, draftLogits, cfg.vocabularySize(), dim); // tied head, no softcap
        w.postProjection.matmul(cur, hNext, bd, dim);
        return draftLogits;
    }

    /**
     * The 1536-dim backbone-space hidden produced by the last {@link #draft} call, to chain
     * depth&gt;1.
     */
    public FloatTensor chainedHidden() {
        return hNext;
    }
}
