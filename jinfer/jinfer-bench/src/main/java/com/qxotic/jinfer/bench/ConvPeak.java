// Microbench: Convolutions.conv1dRows throughput across the register-tile shapes.
//
// Give it a shape census (-Djinfer.convProfile=true on a real synthesis, stderr captured to a file)
// and it benchmarks exactly the shapes that model ran, weighted by the FLOPs the model spent in
// each - so the answer is "which tile makes THIS model fastest", not "which tile wins a shape
// ladder someone made up". Without a census argument it falls back to a generic vocoder ladder.
//
// The tile shape is read once into a static final (Convolutions.TILE_CODE), so ONE JVM measures ONE
// shape - sweep by running the command three times:
//
// Build jinfer-bench, then run this main class in a fresh JVM for each
// -Djinfer.convTile=auto|4x2|4x4 value. Pass the optional census path as the first argument.
//
// Runs on the engine pool, where the vocoder's convolutions actually run
// (Inflect2:438) - the spin pool at physical-core width, not the common pool at logical width.
// Measuring on the common pool would time a dispatch path the model never takes.
package com.qxotic.jinfer.bench;

import com.qxotic.jinfer.RuntimeFlags;
import com.qxotic.jinfer.Segments;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Convolutions;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class ConvPeak {

    /**
     * One benchmarked shape and the work the model spends in it. {@code gflops} is the TOTAL over
     * every call the census saw, so the predicted time for this bucket is {@code gflops / rate}.
     */
    private record Shape(int channels, int time, int kernel, int dilation, double gflops) {

        /** One call's worth of work, which is what a single timed pass does. */
        double gflopsPerCall() {
            return 2.0 * channels * channels * kernel * time / 1e9;
        }

        @Override
        public String toString() {
            return String.format("%4dch x %7dt  k=%-2d d=%-2d", channels, time, kernel, dilation);
        }
    }

    /** Fallback when no census is given: a generic vocoder ladder, times for ~2 s of audio. */
    private static final Shape[] LADDER = {
        new Shape(96, 4232, 11, 1, 1), new Shape(48, 33856, 11, 1, 1),
        new Shape(24, 67712, 11, 1, 1), new Shape(12, 135424, 11, 1, 1),
    };

    /**
     * Warm up until the last WINDOW passes agree within TOL - but never before MIN_WARMUP_NANOS of
     * wall clock have gone into this shape. Stability alone is NOT warmth: interpreted and C1 code
     * are more consistent than C2 code, so a pure agreement check certifies a cold loop as warm and
     * reports its speed. That artifact produced a 35x spread between JVMs on identical work.
     */
    private static final int WINDOW = 3;

    private static final double TOL = 0.03;
    private static final int REPS = 5;
    private static final long MIN_WARMUP_NANOS = 1_500_000_000L;
    private static final long MAX_WARMUP_NANOS = 6_000_000_000L;

    /** Buckets below this share of total FLOPs are skipped - and reported as skipped. */
    private static final double SHARE_FLOOR = 0.002;

    public static void main(String[] args) throws IOException {
        Shape[] shapes = args.length > 0 ? census(Path.of(args[0])) : LADDER;
        double censusTotal = Arrays.stream(shapes).mapToDouble(Shape::gflops).sum();
        // The tile is NOT readable at run time in a native image (Convolutions.TILE_CODE freezes at
        // build), so this echoes the request, not the built binary - label it as such or a sweep
        // reads as if every image were "auto".
        System.out.printf(
                "convTile(requested)=%s  threads=%d  vectorBits=%d  shapes=%d"
                        + "  %.1f GFLOP of model work%n%n",
                System.getProperty("jinfer.convTile", "auto"),
                RuntimeFlags.THREADS,
                Segments.vectorBits(),
                shapes.length,
                censusTotal);
        System.out.printf("%-30s %11s %11s %10s%n", "shape", "rate", "share", "predicted");
        double predicted = 0;
        for (Shape shape : shapes) {
            double[] timings = measure(shape);
            Arrays.sort(timings);
            double rate = shape.gflopsPerCall() / timings[timings.length / 2];
            double seconds = shape.gflops() / rate;
            predicted += seconds;
            System.out.printf(
                    "%-30s %7.0f G/s %10.1f%% %8.3f s%n",
                    shape, rate, 100 * shape.gflops() / censusTotal, seconds);
        }
        // The number that decides the default: how long this model's convolutions take end to end.
        System.out.printf("%n%-30s %31.3f s%n", "TOTAL CONVOLUTION TIME", predicted);
    }

    /**
     * Parse a {@code -Djinfer.convProfile} dump and fold it into one bucket per (channels, kernel,
     * dilation). A bucket keeps the FLOP-weighted mean time, because the same layer runs at several
     * chunk lengths and they behave alike - what differs between buckets is the shape, not the
     * chunking.
     */
    private static Shape[] census(Path path) throws IOException {
        Pattern row =
                Pattern.compile(
                        "\\s*(\\d+)->\\s*(\\d+)"
                                + " ch\\s+(\\d+)t\\s+k=(\\d+)\\s+d=(\\d+)\\s+(\\d+)\\s+([0-9.]+)");
        record Bucket(int channels, int kernel, int dilation) {}
        Map<Bucket, double[]> buckets = new LinkedHashMap<>(); // {gflops, gflops*time}
        for (String line : Files.readAllLines(path)) {
            Matcher m = row.matcher(line);
            if (!m.find()) continue;
            int channels = Integer.parseInt(m.group(2));
            int time = Integer.parseInt(m.group(3));
            double gflops = Double.parseDouble(m.group(7));
            double[] cell =
                    buckets.computeIfAbsent(
                            new Bucket(
                                    channels,
                                    Integer.parseInt(m.group(4)),
                                    Integer.parseInt(m.group(5))),
                            k -> new double[2]);
            cell[0] += gflops;
            cell[1] += gflops * time;
        }
        double total = buckets.values().stream().mapToDouble(cell -> cell[0]).sum();
        List<Shape> shapes = new ArrayList<>();
        double skipped = 0;
        for (var entry : buckets.entrySet()) {
            double gflops = entry.getValue()[0];
            if (gflops / total < SHARE_FLOOR) {
                skipped += gflops;
                continue;
            }
            int time = (int) Math.round(entry.getValue()[1] / gflops);
            shapes.add(
                    new Shape(
                            entry.getKey().channels(),
                            time,
                            entry.getKey().kernel(),
                            entry.getKey().dilation(),
                            gflops));
        }
        shapes.sort((a, b) -> Double.compare(b.gflops(), a.gflops()));
        if (skipped > 0) {
            System.out.printf(
                    "note: %d buckets below %.1f%% skipped, %.2f GFLOP (%.1f%% of the census)%n",
                    buckets.size() - shapes.size(),
                    100 * SHARE_FLOOR,
                    skipped,
                    100 * skipped / total);
        }
        return shapes.toArray(new Shape[0]);
    }

    /** Seconds per pass, REPS of them, after an adaptive warmup. */
    private static double[] measure(Shape shape) {
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            MemoryView<MemorySegment> in =
                    Views.allocateF32(memory, shape.channels(), shape.time());
            MemoryView<MemorySegment> out =
                    Views.allocateF32(memory, shape.channels(), shape.time());
            Random random = new Random(42);
            float[] input = new float[shape.channels() * shape.time()];
            for (int i = 0; i < input.length; i++) input[i] = random.nextFloat() * 2 - 1;
            Views.copyFromArray(in, 0, input, 0, input.length, "input");
            float[] taps = new float[shape.channels() * shape.kernel() * shape.channels()];
            for (int i = 0; i < taps.length; i++) taps[i] = (random.nextFloat() * 2 - 1) * 0.05f;

            double[] recent = new double[WINDOW];
            long warmupStart = System.nanoTime();
            for (int pass = 0; ; pass++) {
                recent[pass % WINDOW] = pass(shape, in, out, taps);
                long spent = System.nanoTime() - warmupStart;
                if (spent >= MAX_WARMUP_NANOS) break;
                if (spent >= MIN_WARMUP_NANOS && pass >= WINDOW && stable(recent)) break;
            }
            double[] timings = new double[REPS];
            for (int rep = 0; rep < REPS; rep++) timings[rep] = pass(shape, in, out, taps);
            return timings;
        }
    }

    private static boolean stable(double[] recent) {
        double low = Double.MAX_VALUE, high = 0;
        for (double t : recent) {
            low = Math.min(low, t);
            high = Math.max(high, t);
        }
        return (high - low) / low <= TOL;
    }

    private static double pass(
            Shape shape,
            MemoryView<MemorySegment> in,
            MemoryView<MemorySegment> out,
            float[] taps) {
        {
            long t0 = System.nanoTime();
            Convolutions.conv1dRows(
                    in,
                    shape.channels(),
                    out,
                    shape.channels(),
                    shape.time(),
                    shape.kernel(),
                    shape.dilation(),
                    taps,
                    null);
            return (System.nanoTime() - t0) / 1e9;
        }
    }
}
