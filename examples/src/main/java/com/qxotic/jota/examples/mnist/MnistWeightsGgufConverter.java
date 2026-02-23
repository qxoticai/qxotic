package com.qxotic.jota.examples.mnist;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGMLType;
import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/** Converts legacy mnist_mlp.bin weights into a tiny GGUF metadata file. */
public final class MnistWeightsGgufConverter {

    private MnistWeightsGgufConverter() {}

    public static void main(String[] args) throws IOException {
        Path input =
                args.length > 0
                        ? Path.of(args[0])
                        : Path.of("examples/src/main/resources/mnist_mlp.bin");
        Path output =
                args.length > 1
                        ? Path.of(args[1])
                        : Path.of("examples/src/main/resources/mnist_mlp.gguf");

        byte[] bytes = Files.readAllBytes(input);
        ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);

        int inputSize = MnistMlp.INPUT_SIZE;
        int hidden1 = MnistMlp.HIDDEN1_SIZE;
        int hidden2 = MnistMlp.HIDDEN2_SIZE;
        int outputSize = MnistMlp.OUTPUT_SIZE;

        float[] w1 = read(buffer, inputSize * hidden1);
        float[] b1 = read(buffer, hidden1);
        float[] w2 = read(buffer, hidden1 * hidden2);
        float[] b2 = read(buffer, hidden2);
        float[] w3 = read(buffer, hidden2 * outputSize);
        float[] b3 = read(buffer, outputSize);

        if (buffer.hasRemaining()) {
            throw new IllegalStateException(
                    "Unexpected trailing bytes in input: " + buffer.remaining());
        }

        Builder builder = Builder.newBuilder();
        builder.putString("general.architecture", "mnist_mlp");
        builder.putUnsignedInteger("general.alignment", 32);
        builder.putUnsignedInteger("mnist.input_size", inputSize);
        builder.putUnsignedInteger("mnist.hidden1_size", hidden1);
        builder.putUnsignedInteger("mnist.hidden2_size", hidden2);
        builder.putUnsignedInteger("mnist.output_size", outputSize);
        builder.putTensor(TensorEntry.create("mnist.w1", new long[] {w1.length}, GGMLType.F32, -1));
        builder.putTensor(TensorEntry.create("mnist.b1", new long[] {b1.length}, GGMLType.F32, -1));
        builder.putTensor(TensorEntry.create("mnist.w2", new long[] {w2.length}, GGMLType.F32, -1));
        builder.putTensor(TensorEntry.create("mnist.b2", new long[] {b2.length}, GGMLType.F32, -1));
        builder.putTensor(TensorEntry.create("mnist.w3", new long[] {w3.length}, GGMLType.F32, -1));
        builder.putTensor(TensorEntry.create("mnist.b3", new long[] {b3.length}, GGMLType.F32, -1));

        GGUF gguf = builder.build();

        Files.deleteIfExists(output);
        GGUF.write(gguf, output);
        try (FileChannel channel =
                FileChannel.open(output, StandardOpenOption.WRITE, StandardOpenOption.READ)) {
            writeTensor(channel, gguf, "mnist.w1", w1);
            writeTensor(channel, gguf, "mnist.b1", b1);
            writeTensor(channel, gguf, "mnist.w2", w2);
            writeTensor(channel, gguf, "mnist.b2", b2);
            writeTensor(channel, gguf, "mnist.w3", w3);
            writeTensor(channel, gguf, "mnist.b3", b3);
        }
        System.out.println("Wrote GGUF weights: " + output);
    }

    private static float[] read(ByteBuffer buffer, int count) {
        float[] out = new float[count];
        for (int i = 0; i < count; i++) {
            out[i] = buffer.getFloat();
        }
        return out;
    }

    private static void writeTensor(FileChannel channel, GGUF gguf, String name, float[] values)
            throws IOException {
        TensorEntry entry = gguf.getTensor(name);
        if (entry == null) {
            throw new IllegalStateException("Missing tensor entry: " + name);
        }
        if (entry.ggmlType() != GGMLType.F32) {
            throw new IllegalStateException("Expected F32 tensor for " + name);
        }
        long expectedBytes = entry.ggmlType().byteSizeForShape(entry.shape());
        long bytes = (long) values.length * Float.BYTES;
        if (expectedBytes != bytes) {
            throw new IllegalStateException(
                    "Tensor size mismatch for "
                            + name
                            + ": expected bytes "
                            + expectedBytes
                            + " got "
                            + bytes);
        }
        ByteBuffer buffer = ByteBuffer.allocate((int) bytes).order(ByteOrder.nativeOrder());
        for (float value : values) {
            buffer.putFloat(value);
        }
        buffer.flip();

        long position = gguf.getTensorDataOffset() + entry.offset();
        while (buffer.hasRemaining()) {
            position += channel.write(buffer, position);
        }
    }
}
