package com.qxotic.jota.examples.mnist;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.impl.DomainFactory;
import com.qxotic.jota.memory.impl.MemoryFactory;
import com.qxotic.jota.tensor.Tensor;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;

final class MnistMlp {
    static final int INPUT_SIZE = 784;
    static final int HIDDEN1_SIZE = 128;
    static final int HIDDEN2_SIZE = 64;
    static final int OUTPUT_SIZE = 10;

    private final Weights weights;

    MnistMlp() {
        Weights hostWeights = Weights.load();
        Device targetDevice = Device.defaultDevice();
        this.weights =
                Device.PANAMA.equals(targetDevice) ? hostWeights : hostWeights.copyTo(targetDevice);
    }

    InferenceResult infer(float[] batch, int batchSize) {
        Tensor input = Tensor.of(batch, Shape.of(batchSize, INPUT_SIZE));
        Tensor logits = forward(input);
        float[] probabilities = toHostArray(softmaxRows(logits));
        return buildResult(probabilities, batchSize);
    }

    InferenceTimings benchmark(float[] batch, int batchSize) {
        long startBatch = System.nanoTime();
        infer(batch, batchSize);
        long batchNs = System.nanoTime() - startBatch;

        long startSeq = System.nanoTime();
        int stride = INPUT_SIZE;
        float[] sample = new float[stride];
        for (int i = 0; i < batchSize; i++) {
            System.arraycopy(batch, i * stride, sample, 0, stride);
            infer(sample, 1);
        }
        long seqNs = System.nanoTime() - startSeq;
        return new InferenceTimings(batchNs, seqNs);
    }

    private Tensor forward(Tensor input) {
        int batchSize = Math.toIntExact(input.shape().flatAt(0));
        return input.matmul(weights.w1())
                .add(expandBias(weights.b1(), batchSize))
                .relu()
                .matmul(weights.w2())
                .add(expandBias(weights.b2(), batchSize))
                .relu()
                .matmul(weights.w3())
                .add(expandBias(weights.b3(), batchSize));
    }

    private static Tensor expandBias(Tensor row, int batchSize) {
        if (batchSize == 1) {
            return row;
        }
        return row.expand(Shape.of(batchSize, row.shape().flatAt(1)));
    }

    private InferenceResult buildResult(float[] probabilities, int batchSize) {
        int[] preds = new int[batchSize];
        float[] confidences = new float[batchSize];
        int outputSize = OUTPUT_SIZE;

        for (int row = 0; row < batchSize; row++) {
            int base = row * outputSize;
            int best = 0;
            float bestVal = probabilities[base];
            for (int col = 1; col < outputSize; col++) {
                float v = probabilities[base + col];
                if (v > bestVal) {
                    bestVal = v;
                    best = col;
                }
            }
            preds[row] = best;
            confidences[row] = bestVal;
        }

        float[] firstProbs = new float[outputSize];
        System.arraycopy(probabilities, 0, firstProbs, 0, outputSize);
        return new InferenceResult(preds, confidences, firstProbs);
    }

    private float[] toHostArray(Tensor tensor) {
        MemoryView<?> src = tensor.materialize();
        int size = Math.toIntExact(src.shape().size());
        float[] out = new float[size];

        @SuppressWarnings("unchecked")
        MemoryDomain<Object> srcDomain =
                (MemoryDomain<Object>)
                        Environment.current().runtimeFor(src.memory().device()).memoryDomain();
        @SuppressWarnings("unchecked")
        MemoryView<Object> srcView = (MemoryView<Object>) src;

        MemoryDomain<float[]> floatDomain = DomainFactory.ofFloats();
        MemoryView<float[]> dstView =
                MemoryView.of(
                        MemoryFactory.ofFloats(out), DataType.FP32, Layout.rowMajor(src.shape()));

        MemoryDomain.copy(srcDomain, srcView, floatDomain, dstView);
        return out;
    }

    private static Tensor softmaxRows(Tensor logits) {
        long batchSize = logits.shape().flatAt(0);
        Tensor exponentials = logits.exp();
        Tensor onesColumn = Tensor.ones(Shape.of(OUTPUT_SIZE, 1));
        Tensor sums = exponentials.matmul(onesColumn);
        return exponentials.divide(sums.expand(Shape.of(batchSize, OUTPUT_SIZE)));
    }

    record Weights(
            Tensor w1, Tensor b1, Tensor w2, Tensor b2, Tensor w3, Tensor b3, Arena mmapArena) {
        static Weights load() {
            Path ggufPath = resolveGgufPath();
            try {
                GGUF gguf = GGUF.read(ggufPath);
                long tensorBase = gguf.getTensorDataOffset();
                Arena arena = Arena.ofAuto();
                try (FileChannel channel = FileChannel.open(ggufPath, StandardOpenOption.READ)) {
                    Tensor w1 =
                            mapTensorToTensor(
                                    gguf,
                                    channel,
                                    arena,
                                    tensorBase,
                                    "mnist.w1",
                                    Shape.of(INPUT_SIZE, HIDDEN1_SIZE));
                    Tensor b1 =
                            mapTensorToTensor(
                                    gguf,
                                    channel,
                                    arena,
                                    tensorBase,
                                    "mnist.b1",
                                    Shape.of(1, HIDDEN1_SIZE));
                    Tensor w2 =
                            mapTensorToTensor(
                                    gguf,
                                    channel,
                                    arena,
                                    tensorBase,
                                    "mnist.w2",
                                    Shape.of(HIDDEN1_SIZE, HIDDEN2_SIZE));
                    Tensor b2 =
                            mapTensorToTensor(
                                    gguf,
                                    channel,
                                    arena,
                                    tensorBase,
                                    "mnist.b2",
                                    Shape.of(1, HIDDEN2_SIZE));
                    Tensor w3 =
                            mapTensorToTensor(
                                    gguf,
                                    channel,
                                    arena,
                                    tensorBase,
                                    "mnist.w3",
                                    Shape.of(HIDDEN2_SIZE, OUTPUT_SIZE));
                    Tensor b3 =
                            mapTensorToTensor(
                                    gguf,
                                    channel,
                                    arena,
                                    tensorBase,
                                    "mnist.b3",
                                    Shape.of(1, OUTPUT_SIZE));
                    return new Weights(w1, b1, w2, b2, w3, b3, arena);
                }
            } catch (IOException e) {
                throw new IllegalStateException("Failed to read mnist_mlp.gguf", e);
            }
        }

        Weights copyTo(Device device) {
            if (w1.device().equals(device)
                    && b1.device().equals(device)
                    && w2.device().equals(device)
                    && b2.device().equals(device)
                    && w3.device().equals(device)
                    && b3.device().equals(device)) {
                return this;
            }
            return new Weights(
                    copyTensorToDevice(w1, device),
                    copyTensorToDevice(b1, device),
                    copyTensorToDevice(w2, device),
                    copyTensorToDevice(b2, device),
                    copyTensorToDevice(w3, device),
                    copyTensorToDevice(b3, device),
                    mmapArena);
        }

        private static Tensor copyTensorToDevice(Tensor source, Device targetDevice) {
            MemoryView<?> src = source.materialize();
            if (src.memory().device().equals(targetDevice)) {
                return source;
            }

            @SuppressWarnings("unchecked")
            MemoryDomain<Object> srcDomain =
                    (MemoryDomain<Object>)
                            Environment.current().runtimeFor(src.memory().device()).memoryDomain();
            @SuppressWarnings("unchecked")
            MemoryDomain<Object> dstDomain =
                    (MemoryDomain<Object>)
                            Environment.current().runtimeFor(targetDevice).memoryDomain();
            @SuppressWarnings("unchecked")
            MemoryView<Object> srcView = (MemoryView<Object>) src;

            Memory<Object> dstMemory =
                    dstDomain.memoryAllocator().allocateMemory(src.dataType(), src.shape().size());
            MemoryView<Object> dstView =
                    MemoryView.of(dstMemory, src.dataType(), Layout.rowMajor(src.shape()));

            MemoryDomain.copy(srcDomain, srcView, dstDomain, dstView);
            return Tensor.of(dstView);
        }

        private static Path resolveGgufPath() {
            URL url = MnistMlp.class.getResource("/mnist_mlp.gguf");
            if (url == null) {
                throw new IllegalStateException("Missing mnist_mlp.gguf resource");
            }
            if ("file".equals(url.getProtocol())) {
                try {
                    return Path.of(url.toURI());
                } catch (URISyntaxException e) {
                    throw new IllegalStateException("Invalid GGUF resource URI", e);
                }
            }

            try (InputStream stream = url.openStream()) {
                Path temp = Files.createTempFile("mnist_mlp_weights", ".gguf");
                Files.copy(stream, temp, StandardCopyOption.REPLACE_EXISTING);
                temp.toFile().deleteOnExit();
                return temp;
            } catch (IOException e) {
                throw new IllegalStateException("Failed to materialize mnist_mlp.gguf", e);
            }
        }

        private static Tensor mapTensorToTensor(
                GGUF gguf,
                FileChannel channel,
                Arena arena,
                long tensorBase,
                String name,
                Shape shape)
                throws IOException {
            TensorEntry tensor = requireTensor(gguf, name);
            long expectedBytes = shape.size() * Float.BYTES;
            validateTensorSize(tensor, name, expectedBytes);

            MemorySegment mapped =
                    channel.map(
                            FileChannel.MapMode.READ_ONLY,
                            tensorBase + tensor.offset(),
                            expectedBytes,
                            arena);
            MemoryView<MemorySegment> view =
                    MemoryView.of(
                            MemoryFactory.ofMemorySegment(mapped),
                            DataType.FP32,
                            Layout.rowMajor(shape));
            Tensor hostMapped = Tensor.of(view);
            return hostMapped;
        }

        private static TensorEntry requireTensor(GGUF gguf, String name) {
            TensorEntry tensor = gguf.getTensor(name);
            if (tensor == null) {
                throw new IllegalStateException("Missing tensor in gguf: " + name);
            }
            if (tensor.ggmlType() != GGMLType.F32) {
                throw new IllegalStateException(
                        "Expected F32 tensor for " + name + " but got " + tensor.ggmlType());
            }
            return tensor;
        }

        private static void validateTensorSize(
                TensorEntry tensor, String name, long expectedBytes) {
            long bytes = tensor.ggmlType().byteSizeForShape(tensor.shape());
            if (bytes != expectedBytes) {
                throw new IllegalStateException(
                        "Unexpected tensor size for "
                                + name
                                + ": expected "
                                + expectedBytes
                                + " bytes, got "
                                + bytes);
            }
        }
    }

    record InferenceResult(int[] preds, float[] confidences, float[] probs) {}

    record InferenceTimings(long batchNs, long seqNs) {
        long batchMs() {
            return batchNs / 1_000_000L;
        }

        long seqMs() {
            return seqNs / 1_000_000L;
        }
    }
}
