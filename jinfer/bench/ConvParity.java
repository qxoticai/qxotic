// Parity check: does the register-tile shape change the numbers conv1dRows produces?
//
// The tile shape is a static final, so ONE JVM sees ONE shape. Run it once per shape and diff the
// output - identical digests mean the tiles are bit-for-bit equivalent, which is what they should
// be (they reorder loads and broadcasts, not the accumulation order within an output vector):
//
//   javac -d /tmp/convpeak --add-modules jdk.incubator.vector \
//         -cp jinfer-core/target/classes bench/ConvParity.java
//   for t in auto 4x2 4x4; do
//     java --enable-preview --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \
//          -Djinfer.convTile=$t -cp /tmp/convpeak:jinfer-core/target/classes \
//          com.qxotic.jinfer.ConvParity > /tmp/parity.$t
//   done
//   diff /tmp/parity.auto /tmp/parity.4x2 && diff /tmp/parity.auto /tmp/parity.4x4
//
// Shapes deliberately include channel counts that are NOT multiples of GROUP=4 and time lengths
// that are not multiples of the vector width, so the tail paths are exercised too.
package com.qxotic.jinfer;

import java.lang.foreign.Arena;
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
                FloatTensor.VECTOR_BIT_SIZE,
                FloatTensor.USE_VECTOR_API);
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
            F32FloatTensor in = F32FloatTensor.allocate(arena, shape.inChannels(), shape.time());
            F32FloatTensor out = F32FloatTensor.allocate(arena, shape.outChannels(), shape.time());
            Random random = new Random(1234);
            for (long i = 0, n = (long) shape.inChannels() * shape.time(); i < n; i++) {
                in.setFloat(i, random.nextFloat() * 2 - 1);
            }
            float[] taps =
                    new float[shape.outChannels() * shape.kernel() * shape.inChannels()];
            for (int i = 0; i < taps.length; i++) taps[i] = (random.nextFloat() * 2 - 1) * 0.1f;
            F32FloatTensor bias = F32FloatTensor.allocate(arena, shape.outChannels());
            for (int i = 0; i < shape.outChannels(); i++) bias.setFloat(i, random.nextFloat() - 0.5f);

            Parallel.onDecodePool(
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
            for (long i = 0, n = (long) shape.outChannels() * shape.time(); i < n; i++) {
                int bits = Float.floatToRawIntBits(out.getFloat(i));
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
