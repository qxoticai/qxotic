package com.qxotic.jinfer.models.parakeet;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.Workspace;
import com.qxotic.jinfer.kernels.Activations;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.MatMul;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

/**
 * Parakeet TDT transducer decoder: the NeMo prediction network (stacked LSTM, PyTorch {@code
 * i,f,g,o} gate order, both biases applied), the joint network (ReLU), and the greedy
 * token-and-duration loop. Matches parakeet.cpp's {@code tdt.cpp} and {@code prediction.cpp}: the
 * first step feeds a zero vector (not the blank embedding), the LSTM state commits only on
 * emission, the inner loop repeats only while the predicted duration is zero, and {@code
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
     * One emitted token: the joint's argmax at {@code frame}, its predicted duration in frames, and
     * its confidence as defined by {@code Transcription.Token}.
     */
    public record Emission(int token, int frame, int duration, double confidence) {}

    /** The stacked LSTM's hidden and cell state. */
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
    private final MemoryView<MemorySegment> embed;
    private final MemoryView<MemorySegment>[] weightInput, weightHidden;
    private final float[][] gateBias;
    private final MemoryView<MemorySegment> encWeight, predWeight, outWeight;
    private final float[] encBias, predBias, outBias;

    @SuppressWarnings("unchecked")
    private ParakeetTdt(Config config, Tensors tensors) {
        this.config = config;
        int hidden = config.predHidden();
        this.embed =
                tensors.vector("decoder.prediction.embed.weight", config.tokenCount() * hidden);
        this.weightInput = new MemoryView[config.predLayers()];
        this.weightHidden = new MemoryView[config.predLayers()];
        this.gateBias = new float[config.predLayers()][];
        String prefix = "decoder.prediction.dec_rnn.lstm.";
        for (int layer = 0; layer < config.predLayers(); layer++) {
            weightInput[layer] = tensors.require(prefix + "weight_ih_l" + layer);
            weightHidden[layer] = tensors.require(prefix + "weight_hh_l" + layer);
            float[] bias = tensors.floats(prefix + "bias_ih_l" + layer, 4 * hidden);
            float[] hiddenBias = tensors.floats(prefix + "bias_hh_l" + layer, 4 * hidden);
            for (int i = 0; i < bias.length; i++) bias[i] += hiddenBias[i];
            gateBias[layer] = bias; // both LSTM biases apply to every step, so they fold
        }
        this.encWeight = tensors.require("joint.enc.weight");
        this.encBias = tensors.floats("joint.enc.bias", config.jointHidden());
        this.predWeight = tensors.require("joint.pred.weight");
        this.predBias = tensors.floats("joint.pred.bias", config.jointHidden());
        this.outWeight = tensors.require("joint.joint_net.2.weight");
        this.outBias = tensors.floats("joint.joint_net.2.bias", config.vPlus());
    }

    Config config() {
        return config;
    }

    static ParakeetTdt load(GGUF gguf, Tensors tensors) {
        if (gguf.getValue(int[].class, "parakeet.tdt.durations") == null)
            throw new IllegalArgumentException(
                    "not a TDT checkpoint: parakeet.tdt.durations absent");
        int encHidden = gguf.getValue(int.class, "parakeet.encoder.d_model");
        return new ParakeetTdt(Config.from(gguf, encHidden), tensors);
    }

    /**
     * The joint's encoder projection for every frame, {@code [frames, jointHidden]}, valid until
     * the workspace's next rewind.
     */
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
     * Blank watchdog. Greedy TDT can get stuck: the prediction state only advances on emission, so
     * a state whose joint prefers blank on every frame mutes the rest of the audio (seen in NeMo
     * itself, where a final '.' silenced the last 58 s of a 90 s clip). After this many frames
     * without an emission the muted span is rescanned, first from the same state, since a later
     * window may hear what an earlier one missed, then from a fresh state, which escapes the trap.
     * If both stay silent the span is taken for silence and decoding moves on.
     */
    private static final int WATCHDOG_FRAMES = 60; // 4.8 s at the 80 ms frame

    /**
     * Where greedy decoding stands, so it can stop at one window's edge and resume in the next: the
     * committed prediction state, the last token (-1 before any, or after a watchdog reset), the
     * next frame to decode, and the watchdog's bookkeeping. Frames are absolute stream positions.
     */
    static final class Cursor {
        private State committed;
        private int lastToken = -1;
        private int frame;
        private int lastEmissionFrame; // where a rescan resumes
        private int quietSince; // the watchdog's clock: the last emission or rescan
        private int rescans; // of the current muted span: at most two

        Cursor(Config config) {
            committed = new State(config.predLayers(), config.predHidden());
        }

        private Cursor(State committed) {
            this.committed = committed;
        }

        /** The next frame to decode. */
        int frame() {
            return frame;
        }

        Cursor copy() {
            Cursor copy =
                    new Cursor(new State(committed.hidden.length, committed.hidden[0].length));
            copy.committed.copyFrom(committed);
            copy.lastToken = lastToken;
            copy.frame = frame;
            copy.lastEmissionFrame = lastEmissionFrame;
            copy.quietSince = quietSince;
            copy.rescans = rescans;
            return copy;
        }
    }

    /** Greedy TDT over all {@code frames} joint-projected frames, from a fresh start. */
    List<Emission> decode(float[] encProjection, int frames, Workspace workspace) {
        return decode(encProjection, 0, frames, new Cursor(config), workspace);
    }

    /**
     * Greedy TDT from {@code cursor} up to, not including, frame {@code until}, advancing the
     * cursor. {@code encProjection} holds the frames from {@code first} on, which must cover the
     * cursor's frame and {@code until}. A predicted duration may carry the cursor past {@code
     * until}; the next call resumes there.
     */
    List<Emission> decode(
            float[] encProjection, int first, int until, Cursor cursor, Workspace workspace) {
        int hidden = config.predHidden();
        int tokenCount = config.tokenCount(), blank = config.blankId();
        int[] durations = config.durations();
        List<Emission> emissions = new ArrayList<>();
        Scratch scratch = new Scratch(workspace, config);
        State stepped = new State(config.predLayers(), hidden);
        float[] prediction = null;
        float[] logits = workspace.floatsAtLeast(config.vPlus());
        int t = cursor.frame;
        while (t < until) {
            if (t - cursor.quietSince >= WATCHDOG_FRAMES) {
                if (cursor.rescans < 2) t = Math.max(first, cursor.lastEmissionFrame + 1);
                if (cursor.rescans > 0) {
                    cursor.committed = new State(config.predLayers(), hidden);
                    cursor.lastToken = -1;
                }
                cursor.rescans = Math.min(cursor.rescans + 1, 2);
                cursor.quietSince = t;
                prediction = null;
            }
            int symbolsAdded = 0;
            boolean needLoop = true;
            int skip = 0;
            while (needLoop && symbolsAdded < config.maxSymbols()) {
                if (prediction == null)
                    prediction = predStep(cursor.lastToken, cursor.committed, stepped, scratch);
                jointLogits(encProjection, t - first, prediction, logits, scratch);
                int token = argmax(logits, 0, tokenCount);
                int durationIndex = argmax(logits, tokenCount, config.vPlus()) - tokenCount;
                skip = durations[durationIndex];
                if (token != blank) {
                    emissions.add(
                            new Emission(token, t, skip, confidence(logits, tokenCount, token)));
                    cursor.lastToken = token;
                    cursor.lastEmissionFrame = t;
                    cursor.quietSince = t;
                    cursor.rescans = 0;
                    cursor.committed.copyFrom(stepped);
                    prediction = null;
                }
                symbolsAdded++;
                t += skip;
                needLoop = skip == 0;
            }
            // parakeet.cpp advances one extra frame when the symbol budget is exhausted, even
            // when the final iteration already advanced.
            if (symbolsAdded == config.maxSymbols()) t += 1;
        }
        cursor.frame = t;
        return emissions;
    }

    /** Bracketed specials ({@code <unk>}, {@code [..]}) carry no text. */
    static boolean isSpecial(String piece) {
        return !piece.isEmpty()
                && ((piece.startsWith("<") && piece.endsWith(">"))
                        || (piece.startsWith("[") && piece.endsWith("]")));
    }

    /** Per-decode buffers for the single-row steps, drawn from the state's workspace. */
    private static final class Scratch {
        final MemoryView<MemorySegment> x, h, zInput, zHidden, fused, logits;
        final float[] zInputArr, zHiddenArr, fusedArr;

        Scratch(Workspace workspace, Config config) {
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
     * One prediction-network step from {@code in} into {@code out}, which must differ, feeding
     * {@code token} or, when it is -1, a zero vector. Returns the top layer's hidden vector, which
     * is {@code out}'s own array.
     */
    float[] predStep(int token, State in, State out, Scratch scratch) {
        int hidden = config.predHidden();
        if (token < 0) Ops.fillInPlace(scratch.x, 0, hidden, 0f);
        else Convert.copyToF32(embed, (long) token * hidden, scratch.x, 0, hidden);
        for (int layer = 0; layer < config.predLayers(); layer++) {
            if (layer > 0)
                Views.copyFromArray(scratch.x, 0, out.hidden[layer - 1], 0, hidden, "lstm input");
            Views.copyFromArray(scratch.h, 0, in.hidden[layer], 0, hidden, "lstm hidden");
            MatMul.gemm(
                    weightInput[layer],
                    scratch.x,
                    hidden,
                    scratch.zInput,
                    4 * hidden,
                    4 * hidden,
                    1,
                    hidden);
            MatMul.gemm(
                    weightHidden[layer],
                    scratch.h,
                    hidden,
                    scratch.zHidden,
                    4 * hidden,
                    4 * hidden,
                    1,
                    hidden);
            Views.copyToArray(scratch.zInput, 0, scratch.zInputArr, 0, 4 * hidden, "lstm gates");
            Views.copyToArray(scratch.zHidden, 0, scratch.zHiddenArr, 0, 4 * hidden, "lstm gates");
            float[] bias = gateBias[layer];
            float[] cell = in.cell[layer];
            for (int c = 0; c < hidden; c++) {
                float inputGate = Activations.sigmoid(z(scratch, bias, c));
                float forgetGate = Activations.sigmoid(z(scratch, bias, hidden + c));
                float candidate = (float) Math.tanh(z(scratch, bias, 2 * hidden + c));
                float outputGate = Activations.sigmoid(z(scratch, bias, 3 * hidden + c));
                float newCell = forgetGate * cell[c] + inputGate * candidate;
                out.cell[layer][c] = newCell;
                out.hidden[layer][c] = outputGate * (float) Math.tanh(newCell);
            }
        }
        return out.hidden[config.predLayers() - 1];
    }

    private static float z(Scratch scratch, float[] bias, int index) {
        return scratch.zInputArr[index] + scratch.zHiddenArr[index] + bias[index];
    }

    /** Raw joint logits: {@code joint_net.2 · relu(encProj[frame] + pred(prediction)) + bias}. */
    void jointLogits(
            float[] encProjection, int frame, float[] prediction, float[] logits, Scratch scratch) {
        int jointHidden = config.jointHidden();
        Views.copyFromArray(scratch.x, 0, prediction, 0, config.predHidden(), "pred output");
        MatMul.gemm(
                predWeight,
                scratch.x,
                config.predHidden(),
                scratch.fused,
                jointHidden,
                jointHidden,
                1,
                config.predHidden());
        float[] fused = scratch.fusedArr;
        Views.copyToArray(scratch.fused, 0, fused, 0, jointHidden, "joint fused");
        int base = frame * jointHidden;
        for (int c = 0; c < jointHidden; c++)
            fused[c] = Math.max(fused[c] + predBias[c] + encProjection[base + c], 0f);
        Views.copyFromArray(scratch.fused, 0, fused, 0, jointHidden, "joint fused");
        MatMul.gemm(
                outWeight,
                scratch.fused,
                jointHidden,
                scratch.logits,
                config.vPlus(),
                config.vPlus(),
                1,
                jointHidden);
        Views.copyToArray(scratch.logits, 0, logits, 0, config.vPlus(), "joint logits");
        for (int v = 0; v < config.vPlus(); v++) logits[v] += outBias[v];
    }

    /** Parity probe: the prediction network's first output (zero input, zero state). */
    float[] probeSos(Workspace workspace) {
        State zero = new State(config.predLayers(), config.predHidden());
        State stepped = new State(config.predLayers(), config.predHidden());
        return predStep(-1, zero, stepped, new Scratch(workspace, config));
    }

    /** Parity probe: raw joint logits for one frame with a given prediction output. */
    float[] probeJointLogits(
            float[] encProjection, int frame, float[] prediction, Workspace workspace) {
        float[] logits = new float[config.vPlus()];
        jointLogits(encProjection, frame, prediction, logits, new Scratch(workspace, config));
        return logits;
    }

    /** The token slice's max probability, rescaled to {@code (N*p - 1)/(N - 1)}. */
    private static double confidence(float[] logits, int tokenCount, int argmax) {
        double sum = 0;
        for (int v = 0; v < tokenCount; v++) sum += Math.exp(logits[v] - logits[argmax]);
        double p = 1.0 / sum;
        return Math.max(0, Math.min(1, (tokenCount * p - 1) / (tokenCount - 1)));
    }

    /** Ties go to the first index, as in the reference. */
    private static int argmax(float[] values, int from, int to) {
        int best = from;
        for (int i = from + 1; i < to; i++) if (values[i] > values[best]) best = i;
        return best;
    }
}
