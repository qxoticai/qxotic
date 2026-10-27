package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.TokenizationPipeline;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;

final class ParallelBenchmarkPipelines {
    private ParallelBenchmarkPipelines() {}

    static Tokenizer from(Tokenizer tokenizer) {
        if (!(tokenizer instanceof TokenizationPipeline)) {
            throw new IllegalStateException(
                    "Benchmark parallel wrapper requires TokenizationPipeline base instance");
        }
        TokenizationPipeline base = (TokenizationPipeline) tokenizer;
        ParallelTokenizationPipeline.Parallelism parallelism =
                ParallelTokenizationPipeline.Parallelism.fromSystemProperties();
        ParallelTokenizationPipeline decoder =
                new ParallelTokenizationPipeline(
                        base.model(),
                        base.normalizer().orElse(null),
                        base.splitter().orElse(null),
                        parallelism);
        return new ChunkedEncode(base, decoder, Integer.getInteger("toknroll.parallel.threads", 0));
    }

    /**
     * Encodes one large text on many threads by cutting it where the pre-tokenization regex cuts
     * anyway, each chunk through the base pipeline's own fast path; decodes with {@link
     * ParallelTokenizationPipeline}.
     *
     * <p>The cut is before an ASCII letter that follows a newline that follows a visible ASCII
     * character. In the tiktoken family of regexes (r50k, cl100k, o200k, Llama 3, Qwen2) no rule
     * matches a newline and then a letter, and the newline ends its piece the same way whether the
     * letter or the end of the chunk comes next: alone, or after the punctuation before it. A
     * whitespace run before the newline would not do, as r50k's {@code \s+(?!\S)} looks past it.
     * The ids digest of the benchmark against the one-thread run tells.
     */
    private static final class ChunkedEncode implements Tokenizer {
        private static final int MIN_CHUNK = 1 << 20;
        private static final int CHUNKS_PER_THREAD = 8;

        private final Tokenizer base;
        private final ParallelTokenizationPipeline decoder;
        private final ForkJoinPool pool;
        private final int threads;

        ChunkedEncode(Tokenizer base, ParallelTokenizationPipeline decoder, int threads) {
            this.base = base;
            this.decoder = decoder;
            this.threads = threads > 0 ? threads : Runtime.getRuntime().availableProcessors();
            this.pool = new ForkJoinPool(this.threads);
        }

        @Override
        public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
            List<int[]> cuts = cuts(text, start, end);
            if (cuts.size() == 1) {
                base.encodeInto(text, start, end, out);
                return;
            }
            List<IntSequence> parts =
                    pool.submit(
                                    () ->
                                            cuts.parallelStream()
                                                    .map(c -> encodeChunk(text, c[0], c[1]))
                                                    .toList())
                            .join();
            for (IntSequence part : parts) {
                out.addAll(part);
            }
        }

        private IntSequence encodeChunk(CharSequence text, int start, int end) {
            IntSequence.Builder local = IntSequence.newBuilder((end - start) / 3);
            base.encodeInto(text, start, end, local);
            return local.build();
        }

        /** The chunks of {@code text[start, end)}, each at least {@link #MIN_CHUNK} chars. */
        private List<int[]> cuts(CharSequence text, int start, int end) {
            int chunks =
                    Math.max(1, Math.min(threads * CHUNKS_PER_THREAD, (end - start) / MIN_CHUNK));
            int step = (end - start) / chunks;
            List<int[]> out = new ArrayList<>(chunks);
            int from = start;
            for (int k = 1; k < chunks; k++) {
                int cut = cutAfter(text, Math.max(from, start + k * step), end);
                if (cut < 0) {
                    break;
                }
                if (cut > from) {
                    out.add(new int[] {from, cut});
                    from = cut;
                }
            }
            out.add(new int[] {from, end});
            return out;
        }

        /** The first cut at or after {@code p}, or -1. */
        private static int cutAfter(CharSequence text, int p, int end) {
            for (int i = Math.max(p, 2); i < end; i++) {
                char c = text.charAt(i);
                char before = text.charAt(i - 2);
                if (text.charAt(i - 1) == '\n'
                        && (c | 0x20) >= 'a'
                        && (c | 0x20) <= 'z'
                        && before > ' '
                        && before < 0x7F) {
                    return i;
                }
            }
            return -1;
        }

        @Override
        public Vocabulary vocabulary() {
            return base.vocabulary();
        }

        @Override
        public float expectedTokensPerChar() {
            return base.expectedTokensPerChar();
        }

        @Override
        public int countTokens(CharSequence text, int start, int end) {
            return base.countTokens(text, start, end);
        }

        @Override
        public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
            return decoder.decodeBytesInto(tokens, tokenStartIndex, out);
        }

        @Override
        public byte[] decodeBytes(IntSequence tokens) {
            return decoder.decodeBytes(tokens);
        }

        @Override
        public String decode(IntSequence tokens) {
            return decoder.decode(tokens);
        }
    }
}
