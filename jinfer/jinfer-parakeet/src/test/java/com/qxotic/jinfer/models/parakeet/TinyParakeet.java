package com.qxotic.jinfer.models.parakeet;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Random;

/**
 * A tiny random-weight Parakeet TDT checkpoint, written as a real GGUF: the real tensor names,
 * layouts and metadata types at toy sizes, so every code path runs on every checkout in
 * milliseconds. It recognizes nothing; tests use it for structure, determinism and plumbing, never
 * for accuracy.
 */
final class TinyParakeet {

    static final int SAMPLE_RATE = 16_000, N_MELS = 16, N_FFT = 64, WIN = 48, HOP = 160;
    static final int D_MODEL = 16, HEADS = 2, FF_DIM = 32, LAYERS = 2, CONV_KERNEL = 5;
    static final int CHANNELS = 4, PRED = 8, PRED_LAYERS = 2, JOINT = 8, MAX_SYMBOLS = 3;
    static final int[] DURATIONS = {0, 1, 2, 3, 4};
    static final String[] PIECES = {"<unk>", "▁he", "llo", "▁world", "[noise]"};

    private static final int ALIGNMENT = 32;

    private TinyParakeet() {}

    /** Writes the checkpoint to {@code file}; the same seed writes the same bytes. */
    static Path write(Path file, long seed) throws IOException {
        Map<String, long[]> tensors = tensors();
        Builder builder =
                Builder.newBuilder()
                        .putString("general.architecture", "parakeet")
                        .putString("general.name", "tiny-parakeet")
                        .putString("parakeet.arch", "tdt")
                        .putUnsignedInteger("parakeet.encoder.feat_in", N_MELS)
                        .putUnsignedInteger("parakeet.encoder.d_model", D_MODEL)
                        .putUnsignedInteger("parakeet.encoder.n_layers", LAYERS)
                        .putUnsignedInteger("parakeet.encoder.n_heads", HEADS)
                        .putUnsignedInteger("parakeet.encoder.ff_dim", FF_DIM)
                        .putUnsignedInteger("parakeet.encoder.conv_kernel", CONV_KERNEL)
                        .putUnsignedInteger("parakeet.encoder.subsampling_factor", 8)
                        .putUnsignedInteger("parakeet.encoder.subsampling_conv_channels", CHANNELS)
                        .putBoolean("parakeet.encoder.xscaling", false)
                        .putUnsignedInteger("parakeet.preprocessor.sample_rate", SAMPLE_RATE)
                        .putUnsignedInteger("parakeet.preprocessor.n_mels", N_MELS)
                        .putUnsignedInteger("parakeet.preprocessor.n_fft", N_FFT)
                        .putUnsignedInteger("parakeet.preprocessor.win_length", WIN)
                        .putUnsignedInteger("parakeet.preprocessor.hop_length", HOP)
                        .putFloat("parakeet.preprocessor.preemph", 0.97f)
                        .putFloat("parakeet.preprocessor.mag_power", 2f)
                        .putFloat("parakeet.preprocessor.log_zero_guard", 5.9604645e-8f)
                        .putUnsignedInteger("parakeet.vocab_size", PIECES.length)
                        .putUnsignedInteger("parakeet.blank_id", PIECES.length)
                        .putArrayOfString("parakeet.tokenizer.pieces", PIECES)
                        .putUnsignedInteger("parakeet.decoder.pred_hidden", PRED)
                        .putUnsignedInteger("parakeet.decoder.pred_rnn_layers", PRED_LAYERS)
                        .putUnsignedInteger("parakeet.joint.joint_hidden", JOINT)
                        .putUnsignedInteger("parakeet.decoding.max_symbols", MAX_SYMBOLS)
                        .putArrayOfInteger("parakeet.tdt.durations", DURATIONS);
        Map<String, Long> offsets = new LinkedHashMap<>();
        long offset = 0;
        for (var tensor : tensors.entrySet()) {
            builder.putTensor(
                    TensorEntry.create(tensor.getKey(), tensor.getValue(), GGMLType.F32, offset));
            offsets.put(tensor.getKey(), offset);
            offset += align(4 * elements(tensor.getValue()));
        }
        GGUF.write(builder.build(), file);
        long dataStart = GGUF.read(file).getTensorDataOffset();
        Random random = new Random(seed);
        try (FileChannel channel = FileChannel.open(file, StandardOpenOption.WRITE)) {
            for (var tensor : tensors.entrySet()) {
                float[] values = values(tensor.getKey(), tensor.getValue(), random);
                ByteBuffer bytes =
                        ByteBuffer.allocate(4 * values.length).order(ByteOrder.LITTLE_ENDIAN);
                bytes.asFloatBuffer().put(values);
                channel.write(bytes, dataStart + offsets.get(tensor.getKey()));
            }
        }
        return file;
    }

    /** Writes a checkpoint into {@code dir} and loads it into {@code arena}. */
    static Parakeet load(Path dir, Arena arena) throws IOException {
        return Fixtures.load(write(dir.resolve("tiny-parakeet.gguf"), 7), arena);
    }

    /** {@code seconds} of seeded Gaussian noise at {@link #SAMPLE_RATE}. */
    static float[] noise(double seconds, long seed) {
        Random random = new Random(seed);
        float[] pcm = new float[(int) (seconds * SAMPLE_RATE)];
        for (int i = 0; i < pcm.length; i++) pcm[i] = (float) (0.3 * random.nextGaussian());
        return pcm;
    }

    /** Every tensor the loaders read, dims in GGUF order (innermost first), as in the real file. */
    private static Map<String, long[]> tensors() {
        Map<String, long[]> t = new LinkedHashMap<>();
        int flattened = CHANNELS * (N_MELS / 8), vocab = PIECES.length + 1;
        t.put("preprocessor.featurizer.window", dims(WIN));
        t.put("preprocessor.featurizer.fb", dims(N_FFT / 2 + 1, N_MELS, 1));
        t.put("encoder.pre_encode.conv.0.weight", dims(3, 3, 1, CHANNELS));
        t.put("encoder.pre_encode.conv.0.bias", dims(CHANNELS));
        for (int stage : new int[] {2, 5}) {
            t.put("encoder.pre_encode.conv." + stage + ".weight", dims(3, 3, 1, CHANNELS));
            t.put("encoder.pre_encode.conv." + stage + ".bias", dims(CHANNELS));
            t.put(
                    "encoder.pre_encode.conv." + (stage + 1) + ".weight",
                    dims(1, 1, CHANNELS, CHANNELS));
            t.put("encoder.pre_encode.conv." + (stage + 1) + ".bias", dims(CHANNELS));
        }
        t.put("encoder.pre_encode.out.weight", dims(flattened, D_MODEL));
        t.put("encoder.pre_encode.out.bias", dims(D_MODEL));
        for (int i = 0; i < LAYERS; i++) {
            String p = "encoder.layers." + i + ".";
            for (String norm :
                    new String[] {
                        "norm_feed_forward1",
                        "norm_self_att",
                        "norm_conv",
                        "norm_feed_forward2",
                        "norm_out"
                    }) {
                t.put(p + norm + ".weight", dims(D_MODEL));
                t.put(p + norm + ".bias", dims(D_MODEL));
            }
            for (String ff : new String[] {"feed_forward1", "feed_forward2"}) {
                t.put(p + ff + ".linear1.weight", dims(D_MODEL, FF_DIM));
                t.put(p + ff + ".linear2.weight", dims(FF_DIM, D_MODEL));
            }
            for (String projection : new String[] {"q", "k", "v", "out", "pos"})
                t.put(p + "self_attn.linear_" + projection + ".weight", dims(D_MODEL, D_MODEL));
            t.put(p + "self_attn.pos_bias_u", dims(D_MODEL / HEADS, HEADS));
            t.put(p + "self_attn.pos_bias_v", dims(D_MODEL / HEADS, HEADS));
            t.put(p + "conv.pointwise_conv1.weight", dims(1, D_MODEL, 2 * D_MODEL));
            t.put(p + "conv.depthwise_conv.weight", dims(CONV_KERNEL, 1, D_MODEL));
            for (String bn : new String[] {"weight", "bias", "running_mean", "running_var"})
                t.put(p + "conv.batch_norm." + bn, dims(D_MODEL));
            t.put(p + "conv.pointwise_conv2.weight", dims(1, D_MODEL, D_MODEL));
        }
        t.put("decoder.prediction.embed.weight", dims(PRED, vocab));
        for (int layer = 0; layer < PRED_LAYERS; layer++) {
            String p = "decoder.prediction.dec_rnn.lstm.";
            t.put(p + "weight_ih_l" + layer, dims(PRED, 4 * PRED));
            t.put(p + "weight_hh_l" + layer, dims(PRED, 4 * PRED));
            t.put(p + "bias_ih_l" + layer, dims(4 * PRED));
            t.put(p + "bias_hh_l" + layer, dims(4 * PRED));
        }
        t.put("joint.pred.weight", dims(PRED, JOINT));
        t.put("joint.pred.bias", dims(JOINT));
        t.put("joint.enc.weight", dims(D_MODEL, JOINT));
        t.put("joint.enc.bias", dims(JOINT));
        t.put("joint.joint_net.2.weight", dims(JOINT, vocab + DURATIONS.length));
        t.put("joint.joint_net.2.bias", dims(vocab + DURATIONS.length));
        return t;
    }

    /**
     * Seeded values that keep activations bounded: a Hann window, a positive filterbank, norms near
     * one, positive variances, small biases, and weights scaled by {@code 1/sqrt(fan-in)}.
     */
    private static float[] values(String name, long[] dims, Random random) {
        float[] v = new float[Math.toIntExact(elements(dims))];
        if (name.endsWith("featurizer.window")) {
            for (int i = 0; i < v.length; i++)
                v[i] = (float) (0.5 - 0.5 * Math.cos(2 * Math.PI * i / (v.length - 1)));
            return v;
        }
        long fanIn = elements(dims) / dims[dims.length - 1];
        for (int i = 0; i < v.length; i++) {
            double g = random.nextGaussian();
            v[i] =
                    (float)
                            (name.endsWith("featurizer.fb")
                                    ? random.nextDouble()
                                    : name.endsWith("running_var")
                                            ? 1 + Math.abs(0.1 * g)
                                            : name.contains("norm") && name.endsWith(".weight")
                                                    ? 1 + 0.1 * g
                                                    : name.endsWith("bias")
                                                                    || name.contains("pos_bias")
                                                                    || name.endsWith("running_mean")
                                                            ? 0.1 * g
                                                            : name.endsWith("embed.weight")
                                                                    ? g
                                                                    : g / Math.sqrt(fanIn));
        }
        return v;
    }

    private static long[] dims(long... dims) {
        return dims;
    }

    private static long elements(long[] dims) {
        long n = 1;
        for (long d : dims) n *= d;
        return n;
    }

    private static long align(long bytes) {
        return (bytes + ALIGNMENT - 1) / ALIGNMENT * ALIGNMENT;
    }
}
