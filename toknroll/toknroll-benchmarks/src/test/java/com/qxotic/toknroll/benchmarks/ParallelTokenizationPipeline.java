package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Normalizer;
import com.qxotic.toknroll.Splitter;
import com.qxotic.toknroll.TokenizationModel;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.ForkJoinPool;

/** Benchmark-only parallel tokenizer pipeline. */
public final class ParallelTokenizationPipeline implements Tokenizer {

    private static final int DEFAULT_MAX_IN_FLIGHT_MULTIPLIER = 2;

    private final TokenizationModel model;
    private final Normalizer normalizer;
    private final Splitter splitter;
    private final boolean hasNormalizer;
    private final boolean hasSplitter;
    private final Parallelism parallelism;
    private final Executor executor;
    private final int effectiveParallelism;

    public ParallelTokenizationPipeline(
            TokenizationModel model,
            Normalizer normalizer,
            Splitter splitter,
            Parallelism parallelism) {
        this.model = Objects.requireNonNull(model, "model");
        this.normalizer = normalizer;
        this.splitter = splitter;
        this.hasNormalizer = normalizer != null;
        this.hasSplitter = splitter != null;
        this.parallelism = Objects.requireNonNull(parallelism, "parallelism");
        this.executor = parallelism.resolveExecutor();
        this.effectiveParallelism = parallelism.resolveParallelism(executor);
    }

    public TokenizationModel model() {
        return model;
    }

    public Optional<Normalizer> normalizer() {
        return Optional.ofNullable(normalizer);
    }

    public Optional<Splitter> splitter() {
        return Optional.ofNullable(splitter);
    }

    public Parallelism parallelism() {
        return parallelism;
    }

    @Override
    public void encodeInto(
            CharSequence text, int startInclusive, int endExclusive, IntSequence.Builder out) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(out, "out");
        validateRange(text, startInclusive, endExclusive);

        if (!hasNormalizer && !hasSplitter) {
            model.encodeInto(text, startInclusive, endExclusive, out);
            return;
        }

        encodeNormalizedInto(normalizeSlice(text, startInclusive, endExclusive), out);
    }

    private void encodeNormalizedInto(CharSequence normalizedText, IntSequence.Builder out) {
        if (!hasSplitter) {
            model.encodeInto(normalizedText, out);
            return;
        }
        if (shouldParallelEncode(normalizedText.length())) {
            encodeNormalizedIntoParallel(normalizedText, out);
            return;
        }
        splitter.splitAll(
                normalizedText,
                0,
                normalizedText.length(),
                (source, chunkStart, chunkEnd) ->
                        model.encodeInto(source, chunkStart, chunkEnd, out));
    }

    @Override
    public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
        Objects.requireNonNull(text, "text");
        validateRange(text, startInclusive, endExclusive);
        if (!hasNormalizer && !hasSplitter) {
            return model.countTokens(text, startInclusive, endExclusive);
        }

        CharSequence current = normalizeSlice(text, startInclusive, endExclusive);
        if (!hasSplitter) {
            return model.countTokens(current, 0, current.length());
        }
        if (shouldParallelEncode(current.length())) {
            return countTokensParallel(current);
        }
        int[] total = {0};
        splitter.splitAll(
                current,
                0,
                current.length(),
                (source, chunkStart, chunkEnd) ->
                        total[0] += model.countTokens(source, chunkStart, chunkEnd));
        return total[0];
    }

    @Override
    public float expectedTokensPerChar() {
        return model.expectedTokensPerChar();
    }

    @Override
    public String decode(IntSequence tokens) {
        Objects.requireNonNull(tokens, "tokens");
        if (!shouldParallelDecode(tokens.length())) {
            return model.decode(tokens);
        }
        return new String(decodeBytes(tokens), StandardCharsets.UTF_8);
    }

    @Override
    public byte[] decodeBytes(IntSequence tokens) {
        Objects.requireNonNull(tokens, "tokens");
        if (!shouldParallelDecode(tokens.length())) {
            return model.decodeBytes(tokens);
        }

        ExecutorCompletionService<DecodeBatchResult> completion =
                new ExecutorCompletionService<DecodeBatchResult>(executor);
        int maxInFlight = Math.max(1, effectiveParallelism * DEFAULT_MAX_IN_FLIGHT_MULTIPLIER);
        int tokenCount = tokens.length();
        int batchSize = adaptiveDecodeBatchSize(tokenCount);

        final int[] submitted = {0};
        final int[] completed = {0};
        final int[] nextBatchId = {0};
        final int[] nextEmitId = {0};
        HashMap<Integer, byte[]> pending = new HashMap<Integer, byte[]>();
        ByteAccumulator out = new ByteAccumulator(Math.min(1 << 20, Math.max(32, tokenCount)));

        for (int start = 0; start < tokenCount; start += batchSize) {
            int end = Math.min(tokenCount, start + batchSize);
            final int chunkStart = start;
            final int chunkEnd = end;
            final int batchId = nextBatchId[0]++;
            completion.submit(
                    () ->
                            new DecodeBatchResult(
                                    batchId,
                                    model.decodeBytes(tokens.subSequence(chunkStart, chunkEnd))));
            submitted[0]++;

            while (submitted[0] - completed[0] >= maxInFlight) {
                DecodeBatchResult result = takeDecodeResult(completion);
                completed[0]++;
                pending.put(result.batchId, result.bytes);
                drainReadyDecodedBatches(out, pending, nextEmitId);
            }
        }

        while (completed[0] < submitted[0]) {
            DecodeBatchResult result = takeDecodeResult(completion);
            completed[0]++;
            pending.put(result.batchId, result.bytes);
            drainReadyDecodedBatches(out, pending, nextEmitId);
        }
        return out.toArray();
    }

    @Override
    public int countBytes(IntSequence tokens) {
        return model.countBytes(tokens);
    }

    @Override
    public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
        return model.decodeBytesInto(tokens, tokenStartIndex, out);
    }

    @Override
    public Vocabulary vocabulary() {
        return model.vocabulary();
    }

    private void validateRange(CharSequence text, int startInclusive, int endExclusive) {
        if (startInclusive < 0 || endExclusive < startInclusive || endExclusive > text.length()) {
            throw new IndexOutOfBoundsException(
                    "Invalid range ["
                            + startInclusive
                            + ", "
                            + endExclusive
                            + ") for text length "
                            + text.length());
        }
    }

    private CharSequence normalizeSlice(CharSequence text, int startInclusive, int endExclusive) {
        CharSequence current =
                (startInclusive == 0 && endExclusive == text.length())
                        ? text
                        : text.subSequence(startInclusive, endExclusive);
        return hasNormalizer ? normalizer.apply(current) : current;
    }

    private boolean shouldParallelEncode(int charLength) {
        return hasSplitter
                && parallelism.enabled()
                && charLength >= parallelism.minEncodeChars()
                && effectiveParallelism > 1;
    }

    private boolean shouldParallelDecode(int tokenLength) {
        return parallelism.enabled()
                && tokenLength >= parallelism.minDecodeTokens()
                && effectiveParallelism > 1;
    }

    private int adaptiveDecodeBatchSize(int tokenCount) {
        int basedOnThreads =
                tokenCount
                        / Math.max(
                                1,
                                effectiveParallelism
                                        * Math.max(1, parallelism.decodeBatchesPerThread()));
        int clamped = Math.max(1024, basedOnThreads);
        return Math.min(tokenCount, clamped);
    }

    private void encodeNormalizedIntoParallel(CharSequence source, IntSequence.Builder out) {
        ExecutorCompletionService<EncodeBatchResult> completion =
                new ExecutorCompletionService<EncodeBatchResult>(executor);
        int maxInFlight = Math.max(1, effectiveParallelism * DEFAULT_MAX_IN_FLIGHT_MULTIPLIER);

        BatchRanges batch = new BatchRanges(source, parallelism.batchChunks());
        final int[] nextBatchId = {0};
        final int[] submitted = {0};
        final int[] completed = {0};
        final int[] nextEmitId = {0};
        HashMap<Integer, IntSequence> pending = new HashMap<Integer, IntSequence>();

        splitter.splitAll(
                source,
                0,
                source.length(),
                (s, chunkStart, chunkEnd) -> {
                    if (chunkStart >= chunkEnd) {
                        return;
                    }
                    batch.add(s, chunkStart, chunkEnd);
                    if (!batch.isFull()) {
                        return;
                    }
                    int batchId = nextBatchId[0]++;
                    completion.submit(batch.newEncodeTask(model, batchId));
                    submitted[0]++;
                    batch.reset();

                    while (submitted[0] - completed[0] >= maxInFlight) {
                        EncodeBatchResult result = takeEncodeResult(completion);
                        completed[0]++;
                        pending.put(Integer.valueOf(result.batchId), result.tokens);
                        drainReadyBatches(out, pending, nextEmitId);
                    }
                });

        if (!batch.isEmpty()) {
            completion.submit(batch.newEncodeTask(model, nextBatchId[0]++));
            submitted[0]++;
        }

        while (completed[0] < submitted[0]) {
            EncodeBatchResult result = takeEncodeResult(completion);
            completed[0]++;
            pending.put(Integer.valueOf(result.batchId), result.tokens);
            drainReadyBatches(out, pending, nextEmitId);
        }
    }

    private int countTokensParallel(CharSequence source) {
        ExecutorCompletionService<Integer> completion =
                new ExecutorCompletionService<Integer>(executor);
        int maxInFlight = Math.max(1, effectiveParallelism * DEFAULT_MAX_IN_FLIGHT_MULTIPLIER);
        BatchRanges batch = new BatchRanges(source, parallelism.batchChunks());
        final int[] submitted = {0};
        final int[] completed = {0};
        final int[] total = {0};

        splitter.splitAll(
                source,
                0,
                source.length(),
                (s, chunkStart, chunkEnd) -> {
                    if (chunkStart >= chunkEnd) {
                        return;
                    }
                    batch.add(s, chunkStart, chunkEnd);
                    if (!batch.isFull()) {
                        return;
                    }
                    completion.submit(batch.newCountTask(model));
                    submitted[0]++;
                    batch.reset();

                    while (submitted[0] - completed[0] >= maxInFlight) {
                        total[0] += takeIntResult(completion);
                        completed[0]++;
                    }
                });

        if (!batch.isEmpty()) {
            completion.submit(batch.newCountTask(model));
            submitted[0]++;
        }

        while (completed[0] < submitted[0]) {
            total[0] += takeIntResult(completion);
            completed[0]++;
        }
        return total[0];
    }

    private static void appendTokens(IntSequence.Builder out, IntSequence tokens) {
        int n = tokens.length();
        out.ensureCapacity(out.size() + n);
        for (int i = 0; i < n; i++) {
            out.add(tokens.intAt(i));
        }
    }

    private static void drainReadyBatches(
            IntSequence.Builder out, HashMap<Integer, IntSequence> pending, int[] nextEmitId) {
        while (true) {
            IntSequence ready = pending.remove(nextEmitId[0]);
            if (ready == null) {
                return;
            }
            appendTokens(out, ready);
            nextEmitId[0]++;
        }
    }

    private static void drainReadyDecodedBatches(
            ByteAccumulator out, HashMap<Integer, byte[]> pending, int[] nextEmitId) {
        while (true) {
            byte[] ready = pending.remove(nextEmitId[0]);
            if (ready == null) {
                return;
            }
            out.addAll(ready);
            nextEmitId[0]++;
        }
    }

    private static EncodeBatchResult takeEncodeResult(
            ExecutorCompletionService<EncodeBatchResult> completion) {
        try {
            return completion.take().get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting for encode task", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            if (cause instanceof RuntimeException) {
                throw (RuntimeException) cause;
            }
            throw new IllegalStateException("Encode task failed", cause);
        }
    }

    private static int takeIntResult(ExecutorCompletionService<Integer> completion) {
        try {
            return completion.take().get().intValue();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting for count task", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            if (cause instanceof RuntimeException) {
                throw (RuntimeException) cause;
            }
            throw new IllegalStateException("Count task failed", cause);
        }
    }

    private static DecodeBatchResult takeDecodeResult(
            ExecutorCompletionService<DecodeBatchResult> completion) {
        try {
            return completion.take().get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("Interrupted while waiting for decode task", e);
        } catch (ExecutionException e) {
            Throwable cause = e.getCause() == null ? e : e.getCause();
            if (cause instanceof RuntimeException) {
                throw (RuntimeException) cause;
            }
            throw new IllegalStateException("Decode task failed", cause);
        }
    }

    private static final class EncodeBatchResult {
        final int batchId;
        final IntSequence tokens;

        EncodeBatchResult(int batchId, IntSequence tokens) {
            this.batchId = batchId;
            this.tokens = tokens;
        }
    }

    private static final class DecodeBatchResult {
        final int batchId;
        final byte[] bytes;

        DecodeBatchResult(int batchId, byte[] bytes) {
            this.batchId = batchId;
            this.bytes = bytes;
        }
    }

    private static final class ByteAccumulator {
        private byte[] data;
        private int size;

        ByteAccumulator(int initialCapacity) {
            this.data = new byte[Math.max(32, initialCapacity)];
        }

        void addAll(byte[] bytes) {
            int n = bytes.length;
            ensureCapacity(size + n);
            System.arraycopy(bytes, 0, data, size, n);
            size += n;
        }

        byte[] toArray() {
            return size == data.length ? data : Arrays.copyOf(data, size);
        }

        private void ensureCapacity(int minCapacity) {
            if (minCapacity <= data.length) {
                return;
            }
            int grown = data.length + (data.length >>> 1) + 1;
            if (grown < minCapacity) {
                grown = minCapacity;
            }
            data = Arrays.copyOf(data, grown);
        }
    }

    private static final class BatchRanges {
        private CharSequence source;
        private int[] starts;
        private int[] ends;
        private int size;

        BatchRanges(CharSequence source, int capacity) {
            this.source = source;
            this.starts = new int[capacity];
            this.ends = new int[capacity];
        }

        void add(CharSequence source, int start, int end) {
            if (this.source != source) {
                this.source = source;
            }
            starts[size] = start;
            ends[size] = end;
            size++;
        }

        boolean isFull() {
            return size == starts.length;
        }

        boolean isEmpty() {
            return size == 0;
        }

        void reset() {
            size = 0;
        }

        Callable<EncodeBatchResult> newEncodeTask(TokenizationModel model, int batchId) {
            final CharSequence sourceRef = source;
            final int[] batchStarts = new int[size];
            final int[] batchEnds = new int[size];
            System.arraycopy(starts, 0, batchStarts, 0, size);
            System.arraycopy(ends, 0, batchEnds, 0, size);
            final int n = size;
            return () -> {
                IntSequence.Builder local = IntSequence.newBuilder(Math.max(8, n * 4));
                for (int i = 0; i < n; i++) {
                    model.encodeInto(sourceRef, batchStarts[i], batchEnds[i], local);
                }
                return new EncodeBatchResult(batchId, local.build());
            };
        }

        Callable<Integer> newCountTask(TokenizationModel model) {
            final CharSequence sourceRef = source;
            final int[] batchStarts = new int[size];
            final int[] batchEnds = new int[size];
            System.arraycopy(starts, 0, batchStarts, 0, size);
            System.arraycopy(ends, 0, batchEnds, 0, size);
            final int n = size;
            return () -> {
                int sum = 0;
                for (int i = 0; i < n; i++) {
                    sum += model.countTokens(sourceRef, batchStarts[i], batchEnds[i]);
                }
                return Integer.valueOf(sum);
            };
        }
    }

    public static final class Parallelism {
        private final boolean enabled;
        private final Executor executor;
        private final int threads;
        private final int minEncodeChars;
        private final int batchChunks;
        private final int minDecodeTokens;
        private final int decodeBatchesPerThread;

        private Parallelism(Builder builder) {
            this.enabled = builder.enabled;
            this.executor = builder.executor;
            this.threads = builder.threads;
            this.minEncodeChars = builder.minEncodeChars;
            this.batchChunks = builder.batchChunks;
            this.minDecodeTokens = builder.minDecodeTokens;
            this.decodeBatchesPerThread = builder.decodeBatchesPerThread;
        }

        public static Builder builder() {
            return new Builder();
        }

        public static Parallelism fromSystemProperties() {
            return builder()
                    .enabled(
                            Boolean.parseBoolean(
                                    System.getProperty("toknroll.parallel.enabled", "true")))
                    .threads(Integer.getInteger("toknroll.parallel.threads", 0))
                    .minEncodeChars(
                            Integer.getInteger(
                                    "toknroll.parallel.minEncodeChars",
                                    Integer.getInteger("toknroll.parallel.minChars", 1 << 14)))
                    .batchChunks(Integer.getInteger("toknroll.parallel.batchChunks", 4096))
                    .minDecodeTokens(
                            Integer.getInteger(
                                    "toknroll.parallel.minDecodeTokens",
                                    Integer.getInteger(
                                            "toknroll.parallel.decode.minTokens", 1 << 16)))
                    .decodeBatchesPerThread(
                            Integer.getInteger("toknroll.parallel.decode.batchesPerThread", 6))
                    .build();
        }

        boolean enabled() {
            return enabled;
        }

        int minEncodeChars() {
            return minEncodeChars;
        }

        int batchChunks() {
            return batchChunks;
        }

        int minDecodeTokens() {
            return minDecodeTokens;
        }

        int decodeBatchesPerThread() {
            return decodeBatchesPerThread;
        }

        Executor resolveExecutor() {
            if (executor != null) {
                return executor;
            }
            if (threads > 1) {
                return new ForkJoinPool(threads);
            }
            return ForkJoinPool.commonPool();
        }

        int resolveParallelism(Executor resolvedExecutor) {
            if (threads > 0) {
                return threads;
            }
            if (resolvedExecutor instanceof ForkJoinPool) {
                return Math.max(1, ((ForkJoinPool) resolvedExecutor).getParallelism());
            }
            return Math.max(1, Runtime.getRuntime().availableProcessors() - 1);
        }

        public static final class Builder {
            private boolean enabled = true;
            private Executor executor;
            private int threads;
            private int minEncodeChars = 1 << 14;
            private int batchChunks = 1024;
            private int minDecodeTokens = 1 << 16;
            private int decodeBatchesPerThread = 6;

            public Builder enabled(boolean enabled) {
                this.enabled = enabled;
                return this;
            }

            public Builder executor(Executor executor) {
                this.executor = executor;
                return this;
            }

            public Builder threads(int threads) {
                this.threads = Math.max(0, threads);
                return this;
            }

            public Builder minEncodeChars(int minEncodeChars) {
                this.minEncodeChars = Math.max(0, minEncodeChars);
                return this;
            }

            public Builder batchChunks(int batchChunks) {
                this.batchChunks = Math.max(1, batchChunks);
                return this;
            }

            public Builder minDecodeTokens(int minDecodeTokens) {
                this.minDecodeTokens = Math.max(0, minDecodeTokens);
                return this;
            }

            public Builder decodeBatchesPerThread(int decodeBatchesPerThread) {
                this.decodeBatchesPerThread = Math.max(1, decodeBatchesPerThread);
                return this;
            }

            public Parallelism build() {
                return new Parallelism(this);
            }
        }
    }
}
