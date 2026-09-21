package com.qxotic.jinfer.models.parakeet;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.Workspace;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Parakeet TDT transducer decoder: the NeMo prediction network (stacked LSTM, PyTorch {@code
 * i,f,g,o} gate order, both biases applied), the joint network (hardcoded ReLU), and the greedy
 * token-and-duration loop. Semantics follow parakeet.cpp {@code tdt.cpp}/{@code prediction.cpp}
 * exactly: the SOS step feeds a zero vector (not the blank embedding), the LSTM state commits only
 * on emission, the inner loop repeats only while the predicted duration is zero, and {@code
 * max_symbols} counts every inner iteration.
 */
public final class ParakeetTdt {

    record Config(
            int vocabSize,
            int blankId,
            int predHidden,
            int predLayers,
            int jointHidden,
            int encHidden,
            int[] durations,
            int maxSymbols,
            String[] pieces) {
        int tokenCount() {
            return vocabSize + 1;
        }

        int vPlus() {
            return tokenCount() + durations.length;
        }

        static Config from(GGUF gguf, int encHidden) {
            return new Config(
                    gguf.getValue(int.class, "parakeet.vocab_size"),
                    gguf.getValue(int.class, "parakeet.blank_id"),
                    gguf.getValue(int.class, "parakeet.decoder.pred_hidden"),
                    gguf.getValue(int.class, "parakeet.decoder.pred_rnn_layers"),
                    gguf.getValue(int.class, "parakeet.joint.joint_hidden"),
                    encHidden,
                    gguf.getValue(int[].class, "parakeet.tdt.durations"),
                    gguf.getValue(int.class, "parakeet.decoding.max_symbols"),
                    gguf.getValue(String[].class, "parakeet.tokenizer.pieces"));
        }
    }

    /**
     * One emitted token: the joint's argmax at {@code frame}, its predicted duration, and the
     * reference's confidence - the max-probability over the token slice rescaled to {@code (N*p -
     * 1)/(N - 1)}.
     */
    public record Emission(int token, int frame, int duration, double confidence) {}

    /** Stacked LSTM carry; commits only on emission. */
    static final class State {
        final float[][] hidden, cell;

        State(int layers, int width) {
            hidden = new float[layers][width];
            cell = new float[layers][width];
        }

        void copyFrom(State other) {
            for (int layer = 0; layer < hidden.length; layer++) {
                System.arraycopy(other.hidden[layer], 0, hidden[layer], 0, hidden[layer].length);
                System.arraycopy(other.cell[layer], 0, cell[layer], 0, cell[layer].length);
            }
        }
    }

    private final Config config;
    private final float[] embed;
    private final MemoryView<MemorySegment>[] weightInput, weightHidden;
    private final float[][] gateBias;
    private final MemoryView<MemorySegment> encWeight, predWeight, outWeight;
    private final float[] encBias, predBias, outBias;

    @SuppressWarnings("unchecked")
    private ParakeetTdt(Config config, Map<String, MemoryView<MemorySegment>> tensors) {
        this.config = config;
        int hidden = config.predHidden();
        this.embed =
                Tensors.floats(
                        tensors,
                        "decoder.prediction.embed.weight",
                        (config.vocabSize() + 1) * hidden);
        this.weightInput = new MemoryView[config.predLayers()];
        this.weightHidden = new MemoryView[config.predLayers()];
        this.gateBias = new float[config.predLayers()][];
        for (int layer = 0; layer < config.predLayers(); layer++) {
            String prefix = "decoder.prediction.dec_rnn.lstm.";
            weightInput[layer] = Tensors.require(tensors, prefix + "weight_ih_l" + layer);
            weightHidden[layer] = Tensors.require(tensors, prefix + "weight_hh_l" + layer);
            float[] inputBias = Tensors.floats(tensors, prefix + "bias_ih_l" + layer, 4 * hidden);
            float[] hiddenBias = Tensors.floats(tensors, prefix + "bias_hh_l" + layer, 4 * hidden);
            gateBias[layer] = new float[4 * hidden];
            for (int i = 0; i < gateBias[layer].length; i++)
                gateBias[layer][i] = inputBias[i] + hiddenBias[i];
        }
        this.encWeight = Tensors.require(tensors, "joint.enc.weight");
        this.encBias = Tensors.floats(tensors, "joint.enc.bias", config.jointHidden());
        this.predWeight = Tensors.require(tensors, "joint.pred.weight");
        this.predBias = Tensors.floats(tensors, "joint.pred.bias", config.jointHidden());
        this.outWeight = Tensors.require(tensors, "joint.joint_net.2.weight");
        this.outBias = Tensors.floats(tensors, "joint.joint_net.2.bias", config.vPlus());
    }

    Config config() {
        return config;
    }

    public static ParakeetTdt load(Path path, Arena arena) throws IOException {
        try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            return load(gguf, ModelLoader.loadTensors(channel, gguf, arena));
        }
    }

    public static ParakeetTdt load(GGUF gguf, Map<String, MemoryView<MemorySegment>> tensors) {
        Objects.requireNonNull(gguf, "gguf");
        Objects.requireNonNull(tensors, "tensors");
        if (gguf.getValue(int[].class, "parakeet.tdt.durations") == null)
            throw new IllegalArgumentException(
                    "not a TDT checkpoint: parakeet.tdt.durations absent");
        int encHidden = gguf.getValue(int.class, "parakeet.encoder.d_model");
        return new ParakeetTdt(Config.from(gguf, encHidden), tensors);
    }

    /**
     * The joint's encoder projection, precomputed for every frame: {@code [frames, jointHidden]}.
     */
    public float[] encProjection(float[] encoderFrameMajor, int frames) {
        MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
        try {
            return encProjection(
                    Views.fromFloatArray(scratch, encoderFrameMajor),
                    frames,
                    new Workspace(scratch));
        } finally {
            Arenas.close(scratch);
        }
    }

    /** As above, from the encoder's own view; the result is the workspace's, until its rewind. */
    float[] encProjection(MemoryView<MemorySegment> encoder, int frames, Workspace workspace) {
        int encHidden = config.encHidden(), jointHidden = config.jointHidden();
        MemoryView<MemorySegment> projected = Views.allocateF32(workspace, frames, jointHidden);
        MatMul.gemm(
                encWeight,
                encoder,
                encHidden,
                projected,
                jointHidden,
                jointHidden,
                frames,
                encHidden);
        float[] result = workspace.floatsAtLeast(frames * jointHidden);
        Views.copyToArray(projected, 0, result, 0, frames * jointHidden, "joint enc projection");
        for (int frame = 0; frame < frames; frame++)
            for (int c = 0; c < jointHidden; c++) result[frame * jointHidden + c] += encBias[c];
        return result;
    }

    /**
     * The blank watchdog: greedy TDT has an absorbing silence state - the prediction state only
     * advances on emission, so a state whose joint prefers blank for every incoming frame mutes the
     * decoder forever (measured on NeMo itself: an utterance-final '.' froze the remaining 58 s of
     * a 90 s clip). After this many consecutive emission-free frames the prediction state resets to
     * the fresh-utterance state, which escapes the trap and is a no-op over genuine silence.
     */
    private static final int WATCHDOG_FRAMES = 60; // 4.8 s at the 80 ms frame

    /** Greedy TDT over {@code frames} joint-projected encoder frames. */
    public List<Emission> decode(float[] encProjection, int frames) {
        MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
        try {
            return decode(encProjection, frames, new Workspace(scratch));
        } finally {
            Arenas.close(scratch);
        }
    }

    /** As above, with every buffer drawn from {@code workspace}. */
    List<Emission> decode(float[] encProjection, int frames, Workspace workspace) {
        int hidden = config.predHidden(), jointHidden = config.jointHidden();
        int tokenCount = config.tokenCount(), blank = config.blankId();
        int[] durations = config.durations();
        List<Emission> emissions = new ArrayList<>();
        Work work = new Work(workspace, config);
        State committed = new State(config.predLayers(), hidden);
        State stepped = new State(config.predLayers(), hidden);
        float[] g = null;
        boolean emittedAny = false;
        int lastToken = -1;
        int lastEmissionFrame = 0;
        int emissionsAtLastFire = -1; // forward progress: one fruitless re-scan, then move on
        float[] logits = workspace.floatsAtLeast(config.vPlus());
        int t = 0;
        while (t < frames) {
            if (t - lastEmissionFrame >= WATCHDOG_FRAMES) {
                committed = new State(config.predLayers(), hidden);
                emittedAny = false;
                g = null;
                // Rewind and re-decode the muted span with the fresh state: over a trap the
                // words come back, over genuine silence the re-scan decodes nothing. A fire
                // with no emissions since the previous fire means the span is truly empty -
                // no second rewind, the loop advances.
                if (emissions.size() != emissionsAtLastFire) {
                    emissionsAtLastFire = emissions.size();
                    t = lastEmissionFrame + 1;
                }
                lastEmissionFrame = t;
            }
            int symbolsAdded = 0;
            boolean needLoop = true;
            int skip = 0;
            while (needLoop && symbolsAdded < config.maxSymbols()) {
                if (g == null)
                    g =
                            predStep(
                                    emittedAny ? lastToken : blank,
                                    !emittedAny,
                                    committed,
                                    stepped,
                                    work);
                jointLogits(encProjection, t, jointHidden, g, logits, work);
                int token = argmax(logits, 0, tokenCount);
                int durationIndex = argmax(logits, tokenCount, config.vPlus()) - tokenCount;
                skip = durations[durationIndex];
                if (token != blank) {
                    emissions.add(
                            new Emission(token, t, skip, confidence(logits, tokenCount, token)));
                    lastToken = token;
                    lastEmissionFrame = t;
                    committed.copyFrom(stepped);
                    emittedAny = true;
                    g = null;
                }
                symbolsAdded++;
                t += skip;
                needLoop = skip == 0;
            }
            // parakeet.cpp advances one extra frame when the symbol budget is exhausted, even
            // when the final iteration already advanced.
            if (symbolsAdded == config.maxSymbols()) t += 1;
        }
        return emissions;
    }

    /** NeMo SentencePiece detokenization with bracketed special tokens dropped. */
    public String text(List<Emission> emissions) {
        StringBuilder joined = new StringBuilder();
        for (Emission emission : emissions) {
            if (emission.token() < 0 || emission.token() >= config.pieces().length) continue;
            String piece = config.pieces()[emission.token()];
            if (isSpecial(piece)) continue;
            joined.append(piece);
        }
        String text = joined.toString().replace('▁', ' ');
        return text.startsWith(" ") ? text.substring(1) : text;
    }

    private static boolean isSpecial(String piece) {
        return !piece.isEmpty()
                && ((piece.startsWith("<") && piece.endsWith(">"))
                        || (piece.startsWith("[") && piece.endsWith("]")));
    }

    /** Per-decode scratch for the gemv-shaped steps, drawn from the state's workspace. */
    private static final class Work {
        final MemoryView<MemorySegment> x, h, zInput, zHidden, fused, logits;
        final float[] zInputArr, zHiddenArr, fusedArr;

        Work(Workspace workspace, Config config) {
            int hidden = config.predHidden();
            x = Views.allocateF32(workspace, 1, hidden);
            h = Views.allocateF32(workspace, 1, hidden);
            zInput = Views.allocateF32(workspace, 1, 4 * hidden);
            zHidden = Views.allocateF32(workspace, 1, 4 * hidden);
            fused = Views.allocateF32(workspace, 1, config.jointHidden());
            logits = Views.allocateF32(workspace, 1, config.vPlus());
            zInputArr = workspace.floatsAtLeast(4 * hidden);
            zHiddenArr = workspace.floatsAtLeast(4 * hidden);
            fusedArr = workspace.floatsAtLeast(config.jointHidden());
        }
    }

    /**
     * One prediction-network step. {@code in} is the committed state; {@code out} receives the
     * stepped state; the return value is the top layer's new hidden vector - {@code out}'s own
     * array, so it lives until {@code out} steps again. {@code in} and {@code out} must differ:
     * each layer reads {@code in} while writing {@code out}.
     */
    float[] predStep(int token, boolean sos, State in, State out, Work work) {
        int hidden = config.predHidden();
        if (sos) Ops.fillInPlace(work.x, 0, hidden, 0f);
        else Views.copyFromArray(work.x, 0, embed, token * hidden, hidden, "lstm input");
        for (int layer = 0; layer < config.predLayers(); layer++) {
            if (layer > 0)
                Views.copyFromArray(work.x, 0, out.hidden[layer - 1], 0, hidden, "lstm input");
            Views.copyFromArray(work.h, 0, in.hidden[layer], 0, hidden, "lstm hidden");
            MatMul.gemm(
                    weightInput[layer],
                    work.x,
                    hidden,
                    work.zInput,
                    4 * hidden,
                    4 * hidden,
                    1,
                    hidden);
            MatMul.gemm(
                    weightHidden[layer],
                    work.h,
                    hidden,
                    work.zHidden,
                    4 * hidden,
                    4 * hidden,
                    1,
                    hidden);
            Views.copyToArray(work.zInput, 0, work.zInputArr, 0, 4 * hidden, "lstm gates");
            Views.copyToArray(work.zHidden, 0, work.zHiddenArr, 0, 4 * hidden, "lstm gates");
            float[] bias = gateBias[layer];
            float[] cell = in.cell[layer];
            for (int c = 0; c < hidden; c++) {
                float inputGate = Activations.sigmoid(z(work, bias, c));
                float forgetGate = Activations.sigmoid(z(work, bias, hidden + c));
                float candidate = (float) Math.tanh(z(work, bias, 2 * hidden + c));
                float outputGate = Activations.sigmoid(z(work, bias, 3 * hidden + c));
                float newCell = forgetGate * cell[c] + inputGate * candidate;
                out.cell[layer][c] = newCell;
                out.hidden[layer][c] = outputGate * (float) Math.tanh(newCell);
            }
        }
        return out.hidden[config.predLayers() - 1];
    }

    private static float z(Work work, float[] bias, int index) {
        return work.zInputArr[index] + work.zHiddenArr[index] + bias[index];
    }

    /** {@code logits = joint_net.2 · relu(encProj[t] + pred_proj(g)) + bias}, raw, no softmax. */
    void jointLogits(
            float[] encProjection,
            int frame,
            int jointHidden,
            float[] g,
            float[] logits,
            Work work) {
        Views.copyFromArray(work.x, 0, g, 0, config.predHidden(), "pred output");
        MatMul.gemm(
                predWeight,
                work.x,
                config.predHidden(),
                work.fused,
                jointHidden,
                jointHidden,
                1,
                config.predHidden());
        float[] fused = work.fusedArr;
        Views.copyToArray(work.fused, 0, fused, 0, jointHidden, "joint fused");
        int base = frame * jointHidden;
        for (int c = 0; c < jointHidden; c++)
            fused[c] = Math.max(fused[c] + predBias[c] + encProjection[base + c], 0f);
        Views.copyFromArray(work.fused, 0, fused, 0, jointHidden, "joint fused");
        MatMul.gemm(
                outWeight,
                work.fused,
                jointHidden,
                work.logits,
                config.vPlus(),
                config.vPlus(),
                1,
                jointHidden);
        Views.copyToArray(work.logits, 0, logits, 0, config.vPlus(), "joint logits");
        for (int v = 0; v < config.vPlus(); v++) logits[v] += outBias[v];
    }

    /** Parity probe: the prediction network's SOS output (zero input, zero state). */
    float[] probeSos() {
        MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
        try {
            Work work = new Work(new Workspace(scratch), config);
            State zero = new State(config.predLayers(), config.predHidden());
            State stepped = new State(config.predLayers(), config.predHidden());
            return predStep(config.blankId(), true, zero, stepped, work);
        } finally {
            Arenas.close(scratch);
        }
    }

    /** Parity probe: raw joint logits for one frame with a given prediction output. */
    float[] probeJointLogits(float[] encProjection, int frame, float[] g) {
        MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
        try {
            Work work = new Work(new Workspace(scratch), config);
            float[] logits = new float[config.vPlus()];
            jointLogits(encProjection, frame, config.jointHidden(), g, logits, work);
            return logits;
        } finally {
            Arenas.close(scratch);
        }
    }

    /** The reference's rescaled max-probability over the token slice (stable softmax). */
    private static double confidence(float[] logits, int tokenCount, int argmax) {
        double sum = 0;
        for (int v = 0; v < tokenCount; v++) sum += Math.exp(logits[v] - logits[argmax]);
        double p = 1.0 / sum;
        return Math.max(0, Math.min(1, (tokenCount * p - 1) / (tokenCount - 1)));
    }

    /** First-index tie-break, matching the reference. */
    private static int argmax(float[] values, int from, int to) {
        int best = from;
        for (int i = from + 1; i < to; i++) if (values[i] > values[best]) best = i;
        return best;
    }
}
