// Inflect2 — VITS-family text-to-waveform model.
//   model = Inflect2.load(Path.of("model.gguf"));
//   Media.Audio audio = model.synthesize(state, tokens, speed, variation, seed);
//
// Supports F16, Q8_0, Q4_0 GGUF; Nano (3.97M), Micro (9.36M) variants.
package com.qxotic.jinfer.models.inflect2;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.*;
import com.qxotic.jinfer.kernels.ModelLoader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public final class Inflect2 {

    static {
        if (System.getProperty("jam.vector.threads") == null)
            System.setProperty("jam.vector.threads", "4");
    }

    // ── Configuration ────────────────────────────────────────────────────

    /**
     * Model dimensions read from GGUF metadata. Both Nano (3.97M params) and Micro (9.36M params)
     * share this layout; the values differ. See {@link Inflect2#load} for how the fields map to
     * GGUF keys.
     */
    public record Configuration(
            int symbolCount,
            int interChannels,
            int hiddenChannels,
            int filterChannels,
            int nHeads,
            int nLayers,
            int kernelSize,
            int sampleRate,
            int upsampleInitialChannel,
            int[] resblockKernelSizes,
            int[] resblockDilationSizes,
            int[] upsampleRates,
            int[] upsampleKernelSizes)
            implements Config {

        @Override
        public int vocabularySize() {
            return symbolCount;
        }

        @Override
        public int contextLength() {
            return 512;
        }
    }

    // ── Weights ───────────────────────────────────────────────────────────

    /**
     * All weight tensors keyed by GGUF name — the only practical layout for 302 heterogeneous
     * tensors (unlike LLMs where every layer shares the same tensor shapes and {@code
     * FloatTensor[]} arrays suffice). A {@code record} following the jinfer model convention. Use
     * {@link #get} for access.
     */
    public record Weights(Map<String, FloatTensor> tensors) {
        public FloatTensor get(String name) {
            FloatTensor t = tensors.get(name);
            if (t == null) throw new IllegalArgumentException("missing tensor: " + name);
            return t;
        }
    }

    // ── State ─────────────────────────────────────────────────────────────

    public static final class State extends BaseState {
        State(Arena arena) {
            super(arena);
        }

        @Override
        public int contextCapacity() {
            return 512;
        }

        @Override
        public int batchCapacity() {
            return 1;
        }

        /**
         * Nothing here carries information across positions: the pools are scratch, fully written
         * before each gemm reads them, so rewinding the cursor is the whole reset.
         */
        @Override
        public void reset() {
            resumeAt(0);
        }

        /** Native tensor pools for gemm — reused across synthesize() calls. */
        F32FloatTensor im2col, out;

        F32FloatTensor getIm2col(int size) {
            if (im2col == null || im2col.size() < size)
                im2col = F32FloatTensor.allocate(arena, size);
            return im2col;
        }

        F32FloatTensor getOut(int size) {
            if (out == null || out.size() < size) out = F32FloatTensor.allocate(arena, size);
            return out;
        }
    }

    // ── Instance ──────────────────────────────────────────────────────────

    private final Configuration cfg;
    private final Weights weights;

    private static final float LEAKY = 0.1f;
    private static final boolean TRACE = Boolean.getBoolean("inflect.debug");
    private static final VectorSpecies<Float> VS = FloatVector.SPECIES_PREFERRED;
    private static final int VW = VS.length();

    private Inflect2(Configuration cfg, Weights weights) {
        this.cfg = cfg;
        this.weights = weights;
    }

    public Configuration config() {
        return cfg;
    }

    public Weights weights() {
        return weights;
    }

    /** GC-managed state memory; {@link #newState(Arena)} to own the lifetime yourself. */
    public State newState() {
        return new State(Arena.ofAuto());
    }

    public State newState(Arena arena) {
        return new State(arena);
    }

    public int sampleRate() {
        return cfg.sampleRate();
    }

    // ── State (set per-synthesize call for conv scratch pools) ──────────

    private State st;

    public static Inflect2 load(Path path) throws IOException {
        try (FileChannel ch = FileChannel.open(path, StandardOpenOption.READ)) {
            return load(ch, path.toString());
        }
    }

    /** Load from a classpath resource (embedded GGUF, e.g. for native image). */
    public static Inflect2 loadResource(String resourcePath) throws IOException {
        var in = Inflect2.class.getResourceAsStream(resourcePath);
        if (in == null) throw new IOException("resource not found: " + resourcePath);
        byte[] bytes = in.readAllBytes();
        Path tmp = Files.createTempFile("inflect2-", ".gguf");
        Files.write(tmp, bytes);
        tmp.toFile().deleteOnExit();
        return load(tmp);
    }

    /**
     * Load from the executable itself — z://default.gguf. The executable must have a ZIP overlay
     * (appended after the ELF) containing STORED GGUF(s). The GGUF header is parsed from bytes read
     * at the ZIP entry offset, and tensor data is mmap'd directly from the executable file — no
     * temp files, no copy of tensor data.
     */
    public static Inflect2 loadSelfArchive(String entryName) throws IOException {
        SelfArchive sa = SelfArchive.open();
        try {
            SelfArchive.Entry e = sa.entry(entryName);

            // Read GGUF header (small: < 64 KB for Inflect2's 302 tensors)
            byte[] headerBytes = sa.readAt(e.offset(), Math.min(e.size(), 65536));
            GGUF gguf = GGUF.read(Channels.newChannel(new ByteArrayInputStream(headerBytes)));

            var raw = ModelLoader.loadTensors(sa.channel(), gguf, e.offset(), Arena.ofAuto());
            Map<String, FloatTensor> map = new HashMap<>();
            for (var entry : raw.entrySet())
                map.put(entry.getKey(), ModelLoader.loadQuantized(entry.getValue()));
            return new Inflect2(readConfig(gguf), new Weights(map));
        } finally {
            sa.channel().close();
            sa.close();
        }
    }

    private static Configuration readConfig(GGUF g) {
        return new Configuration(
                g.getValue(int.class, "inflect.v2.symbol_count"),
                g.getValue(int.class, "inflect.v2.inter_channels"),
                g.getValue(int.class, "inflect.v2.hidden_channels"),
                g.getValue(int.class, "inflect.v2.filter_channels"),
                g.getValue(int.class, "inflect.v2.n_heads"),
                g.getValue(int.class, "inflect.v2.n_layers"),
                g.getValue(int.class, "inflect.v2.kernel_size"),
                g.getValue(int.class, "inflect.v2.sample_rate"),
                g.getValue(int.class, "inflect.v2.upsample_initial_channel"),
                g.getValue(int[].class, "inflect.v2.resblock_kernel_sizes"),
                g.getValue(int[].class, "inflect.v2.resblock_dilation_sizes"),
                g.getValue(int[].class, "inflect.v2.upsample_rates"),
                g.getValue(int[].class, "inflect.v2.upsample_kernel_sizes"));
    }

    private static Inflect2 load(FileChannel ch, String name) throws IOException {
        GGUF g = ModelLoader.readGguf(ch, name);
        var raw = ModelLoader.loadTensors(ch, g, Arena.ofAuto());
        Map<String, FloatTensor> map = new HashMap<>();
        for (var e : raw.entrySet()) map.put(e.getKey(), ModelLoader.loadQuantized(e.getValue()));
        return new Inflect2(readConfig(g), new Weights(map));
    }

    // ═══════════════════════════════════════════════════════════════════════
    // PIPELINE
    // ═══════════════════════════════════════════════════════════════════════

    public Media.Audio synthesize(State st, int[] tok, float speed, float noise, long seed) {
        this.st = st;
        int T = tok.length, H = cfg.hiddenChannels(), L = cfg.interChannels();

        float[] hid = encode(tok, T, H);
        float[] proj =
                conv1d(hid, wt("enc_p.proj.weight"), wt("enc_p.proj.bias"), H, 2 * L, 1, 1, T);
        float[] mean = new float[L * T], lvar = new float[L * T];
        for (int s = 0; s < T; s++)
            for (int c = 0; c < L; c++) {
                mean[c * T + s] = proj[s * 2 * L + c];
                lvar[c * T + s] = proj[s * 2 * L + L + c];
            }

        float[] logDur = duration(hid, T, H);
        int[] dur = new int[T];
        int F = 0;
        for (int s = 0; s < T; s++) {
            dur[s] = Math.max((int) Math.ceil(Math.exp(logDur[s]) * speed), 0);
            F += dur[s];
        }
        F = Math.max(F, 1);
        float[] mE = new float[L * F], lE = new float[L * F];
        int fr = 0;
        for (int s = 0; s < T; s++)
            for (int r = 0; r < dur[s] && fr < F; r++) {
                for (int c = 0; c < L; c++) {
                    mE[c * F + fr] = mean[c * T + s];
                    lE[c * F + fr] = lvar[c * T + s];
                }
                fr++;
            }
        for (; fr < F; fr++)
            for (int c = 0; c < L; c++) {
                mE[c * F + fr] = mean[c * T + T - 1];
                lE[c * F + fr] = lvar[c * T + T - 1];
            }

        Random rng = new Random(seed);
        float[] zC = new float[L * F];
        for (int i = 0; i < L * F; i++)
            zC[i] = mE[i] + (float) rng.nextGaussian() * (float) Math.exp(lE[i]) * noise;
        float[] z = new float[L * F];
        for (int f = 0; f < F; f++) for (int c = 0; c < L; c++) z[f * L + c] = zC[c * F + f];
        trace("z-p", z);

        float[] pcm = decode(flow(z, L, F), L, F);
        return new Media.Audio(pcm, cfg.sampleRate(), 1);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ENCODER
    // ═══════════════════════════════════════════════════════════════════════

    private float[] encode(int[] tok, int T, int H) {
        FloatTensor emb = wt("enc_p.emb.weight");
        int embStride = Math.toIntExact(emb.size() / cfg.symbolCount());
        float[] x = new float[H * T];
        float scl = (float) Math.sqrt(H);
        for (int s = 0; s < T; s++) {
            int off = tok[s] * embStride;
            for (int i = 0; i < H; i++) x[s * H + i] = emb.getFloat(i + off) * scl;
        }
        for (int l = 0; l < cfg.nLayers(); l++) {
            x = transformer(x, T, H, l);
            trace("enc-l" + l, x);
        }
        return x;
    }

    private float[] transformer(float[] x, int T, int H, int l) {
        String b = "enc_p.encoder.";
        float[] attn = mha(x, T, H, l);
        float[] a1 =
                addNorm(
                        x,
                        attn,
                        wt(b + "norm_layers_1." + l + ".gamma"),
                        wt(b + "norm_layers_1." + l + ".beta"),
                        H,
                        T);
        float[] ffn =
                conv1d(
                        a1,
                        wt(b + "ffn_layers." + l + ".conv_1.weight"),
                        wt(b + "ffn_layers." + l + ".conv_1.bias"),
                        H,
                        cfg.filterChannels(),
                        3,
                        1,
                        T);
        relu(ffn);
        float[] ff2 =
                conv1d(
                        ffn,
                        wt(b + "ffn_layers." + l + ".conv_2.weight"),
                        wt(b + "ffn_layers." + l + ".conv_2.bias"),
                        cfg.filterChannels(),
                        H,
                        3,
                        1,
                        T);
        return addNorm(
                a1,
                ff2,
                wt(b + "norm_layers_2." + l + ".gamma"),
                wt(b + "norm_layers_2." + l + ".beta"),
                H,
                T);
    }

    private float[] mha(float[] x, int T, int H, int l) {
        String b = "enc_p.encoder.attn_layers." + l + ".";
        float[] Q = conv1d(x, wt(b + "conv_q.weight"), wt(b + "conv_q.bias"), H, H, 1, 1, T);
        float[] K = conv1d(x, wt(b + "conv_k.weight"), wt(b + "conv_k.bias"), H, H, 1, 1, T);
        float[] V = conv1d(x, wt(b + "conv_v.weight"), wt(b + "conv_v.bias"), H, H, 1, 1, T);
        int nH = cfg.nHeads(), hd = H / nH;
        FloatTensor rk = wt(b + "emb_rel_k"), rv = wt(b + "emb_rel_v");
        float scl = 1f / (float) Math.sqrt(hd);
        float[] out = new float[H * T], scores = new float[T];

        for (int h = 0; h < nH; h++) {
            int co = h * hd;
            for (int qi = 0; qi < T; qi++) {
                float mx = Float.NEGATIVE_INFINITY;
                for (int kj = 0; kj < T; kj++) {
                    float s = 0;
                    for (int d = 0; d < hd; d++) s += Q[qi * H + co + d] * K[kj * H + co + d];
                    s *= scl;
                    int dist = kj - qi;
                    if (dist >= -4 && dist <= 4) {
                        float rs = 0;
                        for (int d = 0; d < hd; d++)
                            rs += Q[qi * H + co + d] * rk.getFloat(d + (dist + 4) * hd);
                        s += rs * scl;
                    }
                    scores[kj] = s;
                    if (s > mx) mx = s;
                }
                float den = 0;
                for (int kj = 0; kj < T; kj++) {
                    scores[kj] = (float) Math.exp(scores[kj] - mx);
                    den += scores[kj];
                }
                for (int kj = 0; kj < T; kj++) {
                    float pv = scores[kj] / den;
                    int dist = kj - qi;
                    for (int d = 0; d < hd; d++) {
                        float val = V[kj * H + co + d];
                        if (dist >= -4 && dist <= 4) val += rv.getFloat(d + (dist + 4) * hd);
                        out[qi * H + co + d] += pv * val;
                    }
                }
            }
        }
        return conv1d(out, wt(b + "conv_o.weight"), wt(b + "conv_o.bias"), H, H, 1, 1, T);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // DURATION + FLOW
    // ═══════════════════════════════════════════════════════════════════════

    private float[] duration(float[] x, int T, int H) {
        float[] h = conv1d(x, wt("dp.conv_1.weight"), wt("dp.conv_1.bias"), H, 256, 3, 1, T);
        relu(h);
        h = norm(h, wt("dp.norm_1.gamma"), wt("dp.norm_1.beta"), 256, T);
        h = conv1d(h, wt("dp.conv_2.weight"), wt("dp.conv_2.bias"), 256, 256, 3, 1, T);
        relu(h);
        h = norm(h, wt("dp.norm_2.gamma"), wt("dp.norm_2.beta"), 256, T);
        return conv1d(h, wt("dp.proj.weight"), wt("dp.proj.bias"), 256, 1, 1, 1, T);
    }

    private float[] flow(float[] z, int L, int F) {
        int half = L / 2;
        for (int fi = 6; fi >= 0; fi -= 2) {
            flip(z, L, F);
            z = coupling(z, half, L, F, fi);
        }
        return z;
    }

    private float[] coupling(float[] z, int half, int L, int F, int fi) {
        String b = "flow.flows." + fi + ".";
        float[] z0 = new float[half * F];
        for (int t = 0; t < F; t++) System.arraycopy(z, t * L, z0, t * half, half);
        int H = cfg.hiddenChannels();
        float[] h = conv1d(z0, wt(b + "pre.weight"), wt(b + "pre.bias"), half, H, 1, 1, F);
        float[] skip = new float[H * F];
        for (int l = 0; l < 4; l++) {
            float[] gates =
                    conv1d(
                            h,
                            wt(b + "enc.in_layers." + l + ".weight"),
                            wt(b + "enc.in_layers." + l + ".bias"),
                            H,
                            2 * H,
                            5,
                            1,
                            F);
            float[] acts = new float[H * F];
            for (int s = 0; s < F; s++)
                for (int c = 0; c < H; c++)
                    acts[s * H + c] =
                            (float) Math.tanh(gates[s * 2 * H + c])
                                    / (1f + (float) Math.exp(-gates[s * 2 * H + H + c]));
            int oc = l < 3 ? 2 * H : H;
            float[] proj =
                    conv1d(
                            acts,
                            wt(b + "enc.res_skip_layers." + l + ".weight"),
                            wt(b + "enc.res_skip_layers." + l + ".bias"),
                            H,
                            oc,
                            1,
                            1,
                            F);
            if (l < 3)
                for (int s = 0; s < F; s++)
                    for (int c = 0; c < H; c++) {
                        h[s * H + c] += proj[s * oc + c];
                        skip[s * H + c] += proj[s * oc + H + c];
                    }
            else for (int i = 0; i < skip.length; i++) skip[i] += proj[i];
        }
        float[] mean = conv1d(skip, wt(b + "post.weight"), wt(b + "post.bias"), H, half, 1, 1, F);
        float[] out = z.clone();
        for (int s = 0; s < F; s++)
            for (int c = 0; c < half; c++) out[s * L + half + c] -= mean[s * half + c];
        return out;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // DECODER
    // ═══════════════════════════════════════════════════════════════════════

    private float[] decode(float[] z, int L, int F) {
        int[] rates = cfg.upsampleRates(),
                kerns = cfg.upsampleKernelSizes(),
                rkerns = cfg.resblockKernelSizes(),
                dils = cfg.resblockDilationSizes();

        float[] x =
                conv1d(
                        z,
                        wt("dec.conv_pre.weight"),
                        wt("dec.conv_pre.bias"),
                        L,
                        cfg.upsampleInitialChannel(),
                        7,
                        1,
                        F);
        trace("dec-pre", x);

        int ch = cfg.upsampleInitialChannel(), T = F;
        for (int stage = 0; stage < 4; stage++) {
            int nCh = ch / 2, rate = rates[stage], upK = kerns[stage];
            for (int i = 0; i < ch * T; i++) {
                float v = x[i];
                x[i] = v > 0 ? v : v * LEAKY;
            }

            int T2 = (T - 1) * rate + upK - 2 * ((upK - rate) / 2);
            x =
                    convT(
                            x,
                            wt("dec.ups." + stage + ".weight"),
                            wt("dec.ups." + stage + ".bias"),
                            ch,
                            nCh,
                            T,
                            T2,
                            upK,
                            rate);
            trace("dec-s" + stage + "-ct", x);

            float[] acc = new float[x.length];
            for (int rb = 0; rb < 3; rb++) {
                int idx = stage * 3 + rb, rk = rkerns[rb];
                float[] branch = x.clone();
                for (int d = 0; d < 3; d++) {
                    int dil = dils[rb * 3 + d];
                    float[] lk = new float[branch.length];
                    for (int i = 0; i < branch.length; i++) {
                        float v = branch[i];
                        lk[i] = v > 0 ? v : v * LEAKY;
                    }
                    float[] xt =
                            conv1dLeaky(
                                    lk,
                                    wt("dec.resblocks." + idx + ".convs1." + d + ".weight"),
                                    wt("dec.resblocks." + idx + ".convs1." + d + ".bias"),
                                    nCh,
                                    nCh,
                                    rk,
                                    dil,
                                    T2,
                                    LEAKY);
                    xt =
                            conv1d(
                                    xt,
                                    wt("dec.resblocks." + idx + ".convs2." + d + ".weight"),
                                    wt("dec.resblocks." + idx + ".convs2." + d + ".bias"),
                                    nCh,
                                    nCh,
                                    rk,
                                    1,
                                    T2);
                    for (int i = 0; i < branch.length; i++) branch[i] += xt[i];
                }
                for (int i = 0; i < acc.length; i++) acc[i] += branch[i];
            }
            for (int i = 0; i < acc.length; i++) x[i] = acc[i] / 3f;
            ch = nCh;
            T = T2;
            trace("dec-s" + stage, x);
        }

        for (int i = 0; i < x.length; i++) {
            float v = x[i];
            x[i] = v > 0 ? v : v * 0.01f;
        }
        x = conv1d(x, wt("dec.conv_post.weight"), null, ch, 1, 7, 1, T);
        for (int i = 0; i < x.length; i++) x[i] = (float) Math.tanh(x[i]);
        return x;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CONVOLUTION PRIMITIVES

    private FloatTensor wt(String name) {
        return weights.get(name);
    }

    private float[] conv1d(
            float[] in,
            FloatTensor wt,
            FloatTensor bias,
            int inC,
            int outC,
            int K,
            int dil,
            int T) {
        return conv1dCore(in, wt, bias, inC, outC, K, dil, T, false, 0f);
    }

    private float[] conv1dLeaky(
            float[] in,
            FloatTensor wt,
            FloatTensor bias,
            int inC,
            int outC,
            int K,
            int dil,
            int T,
            float slope) {
        return conv1dCore(in, wt, bias, inC, outC, K, dil, T, true, slope);
    }

    private float[] conv1dCore(
            float[] in,
            FloatTensor wt,
            FloatTensor bias,
            int inC,
            int outC,
            int K,
            int dil,
            int T,
            boolean leaky,
            float slope) {
        int pad = ((K - 1) * dil) / 2, flat = K * inC;
        int rowStride =
                ((long) flat * outC == wt.size()) ? flat : Math.toIntExact(wt.size() / outC);
        float[] out = new float[outC * T];

        float[] cols = new float[T * rowStride];
        if (K > 1) {
            for (int t = 0; t < T; t++) {
                int dstOff = t * rowStride;
                for (int k = 0; k < K; k++) {
                    int st = t + k * dil - pad;
                    if (st >= 0 && st < T) {
                        for (int ic = 0; ic < inC; ic++)
                            cols[dstOff + k + ic * K] = in[st * inC + ic];
                    }
                }
            }
        } else {
            for (int t = 0; t < T; t++) System.arraycopy(in, t * inC, cols, t * rowStride, inC);
        }

        F32FloatTensor im2col = st.getIm2col(T * rowStride);
        im2col.copyRawFrom(MemorySegment.ofArray(cols), 0, 0, T * rowStride);
        F32FloatTensor on = st.getOut(outC * T);
        wt.gemm(im2col, rowStride, on, outC, T, outC, rowStride, 0);

        for (int i = 0; i < out.length; i++) {
            int oc = i % outC;
            float bv = bias != null ? bias.getFloat(oc) : 0f;
            float val = on.getFloat(i) + bv;
            out[i] = leaky ? (val > 0 ? val : val * slope) : val;
        }
        return out;
    }

    private float[] convT(
            float[] in,
            FloatTensor wt,
            FloatTensor bias,
            int inC,
            int outC,
            int T,
            int T2,
            int K,
            int stride) {
        int pad = (K - stride) / 2, flat = K * inC;
        boolean quant = wt.type().isQuantized();
        int rowStride = quant ? Math.toIntExact(wt.size() / outC) : flat;
        float[] out = new float[outC * T2];
        float[] wk = new float[flat];
        int loopBound = VW >= inC ? 0 : (inC / VW) * VW;

        for (int oc = 0; oc < outC; oc++) {
            if (quant) {
                wt.copyRow(oc * (long) rowStride, wk, 0, flat);
            } else {
                for (int k = 0; k < K; k++)
                    for (int ic = 0; ic < inC; ic++)
                        wk[k * inC + ic] = wt.getFloat(k + oc * K + ic * K * outC);
            }
            float bv = bias != null ? bias.getFloat(oc) : 0f;

            for (int op = 0; op < T2; op++) {
                float sum = bv;
                for (int k = 0; k < K; k++) {
                    int t = (op + pad - k) / stride;
                    if (t < 0 || t >= T || (op + pad - k) % stride != 0) continue;
                    int inpOff = t * inC, wkOff = k * inC;
                    int i = 0;
                    FloatVector acc = FloatVector.zero(VS);
                    for (; i < loopBound; i += VW) {
                        acc =
                                FloatVector.fromArray(VS, wk, wkOff + i)
                                        .fma(FloatVector.fromArray(VS, in, inpOff + i), acc);
                    }
                    float dot = acc.reduceLanes(VectorOperators.ADD);
                    for (; i < inC; i++) dot += wk[wkOff + i] * in[inpOff + i];
                    sum += dot;
                }
                out[op * outC + oc] = sum;
            }
        }
        return out;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // UTILITIES
    // ═══════════════════════════════════════════════════════════════════════

    private static float[] addNorm(
            float[] x, float[] y, FloatTensor gamma, FloatTensor beta, int C, int T) {
        float[] out = new float[C * T];
        for (int s = 0; s < T; s++)
            for (int c = 0; c < C; c++) out[s * C + c] = x[s * C + c] + y[s * C + c];
        return norm(out, gamma, beta, C, T);
    }

    private static float[] norm(float[] x, FloatTensor gamma, FloatTensor beta, int C, int T) {
        float[] out = new float[C * T];
        Norms.layerNorm(out, x, gamma, beta, C, T, 1e-5f);
        return out;
    }

    private static void relu(float[] x) {
        for (int i = 0; i < x.length; i++) {
            float v = x[i];
            x[i] = v > 0 ? v : 0;
        }
    }

    private static void flip(float[] x, int C, int T) {
        for (int s = 0; s < T; s++)
            for (int c = 0; c < C / 2; c++) {
                int a = s * C + c, b = s * C + (C - 1 - c);
                float tmp = x[a];
                x[a] = x[b];
                x[b] = tmp;
            }
    }

    private static void trace(String l, float[] a) {
        if (!TRACE) return;
        float mn = Float.POSITIVE_INFINITY, mx = Float.NEGATIVE_INFINITY;
        for (float v : a) {
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        System.out.printf("[trace] %-14s len=%d [%+.4f, %+.4f]%n", l, a.length, mn, mx);
    }
}
