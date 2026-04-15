package com.qxotic.toknroll.benchmarks;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Tokenizers;
import com.qxotic.toknroll.impl.ClassicBPE;
import java.nio.file.Path;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

/**
 * JMH Benchmarks for tokenizer operations with detailed metrics.
 *
 * <p>This benchmark class provides: - Operations per second (ops/s) - Characters per second
 * (chars/s) - calculated as ops/s × text_length - Tokens per second (tokens/s) - calculated as
 * ops/s × token_count
 *
 * <p>Usage: mvn exec:java
 * -Dexec.mainClass="com.qxotic.tokenizers.benchmarks.TokenizerBenchmarkRunner"
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Thread)
@Fork(value = 1, warmups = 1)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
public class TokenizerBenchmark {

    private static final String CL100K_BASE_HASH =
            "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7";

    private static final String CL100K_PATTERN =
            "'(?i:[sdmt]|ll|ve|re)|[^\\r"
                    + "\\n"
                    + "\\p{L}\\p{N}]?+\\p{L}++|\\p{N}{1,3}+| ?[^\\s\\p{L}\\p{N}]++[\\r"
                    + "\\n"
                    + "]*+|\\s++$|\\s*[\\r"
                    + "\\n"
                    + "]|\\s+(?!\\S)|\\s";

    @Param({"small", "medium", "large"})
    private String textSize;

    @Param({"natural", "code", "mixed"})
    private String textType;

    private Tokenizer tokenizer;
    private String text;
    private IntSequence encodedTokens;

    // Metrics for post-processing
    public int textLength;
    public int tokenCount;

    @Setup(Level.Trial)
    public void setup() {
        tokenizer = createCL100KTokenizer();
        text = generateText(textSize, textType);
        textLength = text.length();
        encodedTokens = tokenizer.encode(text);
        tokenCount = encodedTokens.length();
    }

    private Tokenizer createCL100KTokenizer() {
        try {
            Path tiktokenPath =
                    Path.of(
                            TokenizerBenchmark.class
                                    .getClassLoader()
                                    .getResource("tiktoken/cl100k_base.tiktoken")
                                    .toURI());

            var mergeableRanks =
                    ClassicBPE.loadMergeableRanks(tiktokenPath.toString(), CL100K_BASE_HASH);

            return Tokenizers.tikToken(
                    mergeableRanks,
                    java.util.Map.of(
                            "<|endoftext|>", 100257,
                            "<|fim_prefix|>", 100258,
                            "<|fim_middle|>", 100259,
                            "<|fim_suffix|>", 100260,
                            "<|endofprompt|>", 100276),
                    CL100K_PATTERN);
        } catch (Exception e) {
            throw new RuntimeException("Failed to create tokenizer", e);
        }
    }

    private String generateText(String size, String type) {
        String baseText =
                switch (type) {
                    case "natural" -> generateNaturalLanguage();
                    case "code" -> generateCode();
                    case "mixed" -> generateMixed();
                    default -> throw new IllegalArgumentException("Unknown text type: " + type);
                };

        return switch (size) {
            case "small" -> baseText.substring(0, Math.min(100, baseText.length()));
            case "medium" -> baseText.substring(0, Math.min(1000, baseText.length()));
            case "large" ->
                    baseText.repeat(10).substring(0, Math.min(10000, baseText.length() * 10));
            default -> throw new IllegalArgumentException("Unknown text size: " + size);
        };
    }

    private String generateNaturalLanguage() {
        return """
        Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor
        incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud
        exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure
        dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.
        Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt
        mollit anim id est laborum. Sed ut perspiciatis unde omnis iste natus error sit
        voluptatem accusantium doloremque laudantium, totam rem aperiam, eaque ipsa quae ab
        illo inventore veritatis et quasi architecto beatae vitae dicta sunt explicabo. Nemo
        enim ipsam voluptatem quia voluptas sit aspernatur aut odit aut fugit, sed quia
        consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro
        quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit,
        sed quia non numquam eius modi tempora incidunt ut labore et dolore magnam aliquam
        quaerat voluptatem. Ut enim ad minima veniam, quis nostrum exercitationem ullam
        corporis suscipit laboriosam, nisi ut aliquid ex ea commodi consequatur? Quis autem
        vel eum iure reprehenderit qui in ea voluptate velit esse quam nihil molestiae
        consequatur, vel illum qui dolorem eum fugiat quo voluptas nulla pariatur? At vero
        eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium
        voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint
        occaecati cupiditate non provident, similique sunt in culpa qui officia deserunt
        mollitia animi, id est laborum et dolorum fuga. Et harum quidem rerum facilis est
        et expedita distinctio. Nam libero tempore, cum soluta nobis est eligendi optio
        cumque nihil impedit quo minus id quod maxime placeat facere possimus, omnis voluptas
        assumenda est, omnis dolor repellendus. Temporibus autem quibusdam et aut officiis
        debitis aut rerum necessitatibus saepe eveniet ut et voluptates repudiandae sint
        et molestiae non recusandae. Itaque earum rerum hic tenetur a sapiente delectus,
        ut aut reiciendis voluptatibus maiores alias consequatur aut perferendis doloribus
        asperiores repellat.\
        """;
    }

    private String generateCode() {
        return """
        public class TokenizerExample {
            private final Tokenizer tokenizer;

            public TokenizerExample(Tokenizer tokenizer) {
                this.tokenizer = tokenizer;
            }

            public IntSequence encode(String text) {
                return tokenizer.encode(text);
            }

            public String decode(IntSequence tokens) {
                return tokenizer.decode(tokens);
            }

            public int countTokens(String text) {
                return tokenizer.countTokens(text);
            }

            public static void main(String[] args) {
                Tokenizer tokenizer = createTokenizer();
                TokenizerExample example = new TokenizerExample(tokenizer);

                String text = "Hello, world!";
                IntSequence tokens = example.encode(text);
                System.out.println("Tokens: " + tokens);

                String decoded = example.decode(tokens);
                System.out.println("Decoded: " + decoded);

                int count = example.countTokens(text);
                System.out.println("Token count: " + count);
            }
        }

        class TokenizerUtils {
            public static Tokenizer createTokenizer() {
                return null;
            }

            public static double calculateEfficiency(String text, int tokenCount) {
                return (double) text.length() / tokenCount;
            }
        }

        interface TokenProcessor {
            IntSequence process(String input);
            String reverseProcess(IntSequence tokens);
        }

        enum TokenizerType {
            GPT2("r50k_base"),
            GPT4("cl100k_base"),
            GPT4O("o200k_base");

            private final String encoding;

            TokenizerType(String encoding) {
                this.encoding = encoding;
            }

            public String getEncoding() {
                return encoding;
            }
        }\
        """;
    }

    private String generateMixed() {
        return """
        API Documentation - Tokenizer Service

        The tokenizer service provides efficient text tokenization for NLP applications.

        ## Usage Example

        ```java
        Tokenizer tokenizer = TokenizerFactory.create("cl100k_base");
        String text = "Hello, world! Processing 1,000 tokens per second.";
        IntSequence tokens = tokenizer.encode(text);
        System.out.println("Encoded " + tokens.length() + " tokens");
        ```

        ## Performance Metrics

        Based on our benchmarks:
        - Small texts (<100 chars): ~50,000 ops/sec
        - Medium texts (1K chars): ~10,000 ops/sec
        - Large texts (10K chars): ~1,000 ops/sec

        ## Supported Encodings

        1. r50k_base - GPT-2 compatible (50,257 tokens)
        2. cl100k_base - GPT-4 compatible (100,277 tokens)
        3. o200k_base - GPT-4o compatible (200,019 tokens)

        ## Error Handling

        The tokenizer handles various edge cases:
        - Empty strings: Returns empty sequence
        - Unicode text: Full UTF-8 support
        - Special tokens: Configurable handling

        Contact support@example.com for assistance.\
        """;
    }

    @Benchmark
    public void benchmarkEncode(Blackhole blackhole) {
        IntSequence result = tokenizer.encode(text);
        blackhole.consume(result);
    }

    @Benchmark
    public void benchmarkDecode(Blackhole blackhole) {
        String result = tokenizer.decode(encodedTokens);
        blackhole.consume(result);
    }

    @Benchmark
    public void benchmarkCountTokens(Blackhole blackhole) {
        int result = tokenizer.countTokens(text);
        blackhole.consume(result);
    }
}
