// Parity check: does the register-tile shape change the numbers conv1dRows produces?
//
// The tile shape is a static final, so ONE JVM sees ONE shape. Run it once per shape and diff the
// output - identical digests mean the tiles are bit-for-bit equivalent, which is what they should
// be (they reorder loads and broadcasts, not the accumulation order within an output vector):
//
// Build jinfer-bench, then run this main class in a fresh JVM for each
// -Djinfer.convTile=auto|4x2|4x4 value and diff the outputs:
//
//   diff /tmp/parity.auto /tmp/parity.4x2 && diff /tmp/parity.auto /tmp/parity.4x4
//
// Shapes deliberately include channel counts that are NOT multiples of GROUP=4 and time lengths
// that are not multiples of the vector width, so the tail paths are exercised too.
package com.qxotic.jinfer.bench;

import com.qxotic.jinfer.PanamaMemoryArena;
import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.Segments;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Convolutions;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Random;

public final class ConvParity {

    private record Shape(int inChannels, int outChannels, int time, int kernel, int dilation) {}

    private static final Shape[] SHAPES = {
        new Shape(12, 12, 257, 11, 1),
        new Shape(16, 16, 1024, 3, 1),
        new Shape(16, 16, 1024, 3, 9),
        new Shape(13, 7, 333, 5, 2), // neither channel count is a multiple of GROUP
        new Shape(64, 64, 4801, 7, 1), // time is not a multiple of any vector width
        new Shape(1, 9, 128, 3, 1), // single input channel, ragged output group
    };

    public static void main(String[] args) throws NoSuchAlgorithmException {
        System.out.printf(
                "convTile=%s  vectorBits=%d  vectorApi=%s%n",
                System.getProperty("jinfer.convTile", "auto"),
                Segments.vectorBits(),
                Segments.USE_VECTOR_API);
        for (Shape shape : SHAPES) {
            System.out.printf(
                    "%3d->%3d ch  %6dt  k=%-2d d=%-2d  %s%n",
                    shape.inChannels(),
                    shape.outChannels(),
                    shape.time(),
                    shape.kernel(),
                    shape.dilation(),
                    digest(shape));
        }
    }

    /** SHA-256 over the output's raw float bits - any numerical difference shows up here. */
    private static String digest(Shape shape) throws NoSuchAlgorithmException {
        try (Arena arena = Arena.ofConfined()) {
            var memory = new PanamaMemoryArena(arena);
            MemoryView<MemorySegment> in =
                    Views.allocateF32(memory, shape.inChannels(), shape.time());
            MemoryView<MemorySegment> out =
                    Views.allocateF32(memory, shape.outChannels(), shape.time());
            Random random = new Random(1234);
            float[] input = new float[shape.inChannels() * shape.time()];
            for (int i = 0; i < input.length; i++) input[i] = random.nextFloat() * 2 - 1;
            Views.copyFromArray(in, 0, input, 0, input.length, "input");
            float[] taps = new float[shape.outChannels() * shape.kernel() * shape.inChannels()];
            for (int i = 0; i < taps.length; i++) taps[i] = (random.nextFloat() * 2 - 1) * 0.1f;
            MemoryView<MemorySegment> bias = Views.allocateF32(memory, shape.outChannels());
            float[] biases = new float[shape.outChannels()];
            for (int i = 0; i < biases.length; i++) biases[i] = random.nextFloat() - 0.5f;
            Views.copyFromArray(bias, 0, biases, 0, biases.length, "bias");

            Parallel.runDecodeStep(
                    () -> {
                        Convolutions.conv1dRows(
                                in,
                                shape.inChannels(),
                                out,
                                shape.outChannels(),
                                shape.time(),
                                shape.kernel(),
                                shape.dilation(),
                                taps,
                                bias);
                        return null;
                    });

            MessageDigest sha = MessageDigest.getInstance("SHA-256");
            byte[] four = new byte[4];
            for (float value : Views.toFloatArray(out, "output")) {
                int bits = Float.floatToRawIntBits(value);
                four[0] = (byte) bits;
                four[1] = (byte) (bits >>> 8);
                four[2] = (byte) (bits >>> 16);
                four[3] = (byte) (bits >>> 24);
                sha.update(four);
            }
            return HexFormat.of().formatHex(sha.digest()).substring(0, 32);
        }
    }
}
