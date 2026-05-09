package com.qxotic.toknroll;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.regex.Pattern;

/**
 * Code snippets for Tok'n'Roll documentation.
 *
 * <p>Snippet markers use pymdownx.snippets format: {@code [start:name]} / {@code [end:name]}.
 */
@SuppressWarnings("unused")
class Snippets {

    void zeroAllocEncode() {
        // --8<-- [start:zero-alloc-encode]
        Tokenizer tokenizer = null; // ...
        CharSequence input = "Hello, world!";
        IntSequence.Builder out = IntSequence.newBuilder(64);
        tokenizer.encodeInto(input, 0, input.length(), out);
        IntSequence ids = out.build();
        // --8<-- [end:zero-alloc-encode]
    }

    void zeroAllocDecode() {
        // --8<-- [start:zero-alloc-decode]
        Tokenizer tokenizer = null; // ...
        IntSequence ids = tokenizer.encode("Hello, world!");
        ByteBuffer buf = ByteBuffer.allocate(1024);
        int bytesConsumed = tokenizer.decodeBytesInto(ids, 0, buf);
        buf.flip();
        String text = StandardCharsets.UTF_8.decode(buf).toString();
        // --8<-- [end:zero-alloc-decode]
    }

    void intSequenceBuilder() {
        // --8<-- [start:intsequence-builder]
        IntSequence.Builder builder = IntSequence.newBuilder(64);
        builder.add(1);
        builder.add(2);
        builder.add(50256);
        IntSequence ids = builder.build();
        // --8<-- [end:intsequence-builder]
    }

    void intSequenceOps() {
        // --8<-- [start:intsequence-ops]
        IntSequence seq = IntSequence.of(1, 2, 3, 4, 5);
        int first = seq.intAt(0);
        int len = seq.length();
        IntSequence sub = seq.subSequence(1, 3);
        boolean starts = seq.startsWith(IntSequence.of(1, 2));
        int idx = seq.indexOf(3);
        // --8<-- [end:intsequence-ops]
    }

    void intSequenceIterate() {
        // --8<-- [start:intsequence-iterate]
        IntSequence seq = IntSequence.of(1, 2, 3);
        seq.forEachInt((int value) -> System.out.println(value));
        int[] array = seq.toArray();
        // --8<-- [end:intsequence-iterate]
    }

    void splitterRegex() {
        // --8<-- [start:splitter-regex]
        Splitter splitter =
                Splitter.regex(
                        Pattern.compile(
                                "(?i:'(?:[sdmt]|ll|ve|re)|[^\\r"
                                    + "\\n"
                                    + "\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r"
                                    + "\\n"
                                    + "]*|\\s*[\\r"
                                    + "\\n"
                                    + "]+|\\s+(?!\\S)|\\s+)"));
        // --8<-- [end:splitter-regex]
    }

    void splitterCompose() {
        // --8<-- [start:splitter-compose]
        Splitter firstSplitter = Splitter.identity();
        Splitter secondSplitter = Splitter.identity();
        Splitter composed = Splitter.sequence(firstSplitter, secondSplitter);
        // --8<-- [end:splitter-compose]
    }

    void normalizerCompose() {
        // --8<-- [start:normalizer-compose]
        Normalizer norm =
                Normalizer.sequence(
                        Normalizer.unicode(java.text.Normalizer.Form.NFKC), Normalizer.lowercase());
        // --8<-- [end:normalizer-compose]
    }

    void normalizerVariants() {
        // --8<-- [start:normalizer-variants]
        Normalizer nfc = Normalizer.unicode(java.text.Normalizer.Form.NFC);
        Normalizer lower = Normalizer.lowercase();
        Normalizer none = Normalizer.identity();
        // --8<-- [end:normalizer-variants]
    }

    void pipelineCompose() {
        // --8<-- [start:pipeline-compose]
        Normalizer normalizer = Normalizer.identity();
        Splitter splitter = Splitter.identity();
        TokenizationModel model = null; // ...
        TokenizationPipeline pipeline = Toknroll.pipeline(normalizer, splitter, model);
        Tokenizer innerModel = pipeline.model();
        Optional<Normalizer> innerNorm = pipeline.normalizer();
        Optional<Splitter> innerSplitter = pipeline.splitter();
        // --8<-- [end:pipeline-compose]
    }

    void specialsEncode() {
        // --8<-- [start:specials-encode]
        Tokenizer tokenizer = null; // loaded from HF or GGUF
        Specials specials =
                Specials.compile(
                        tokenizer.vocabulary(),
                        Set.of("<|endoftext|>", "<|im_start|>", "<|im_end|>"));
        IntSequence ids = specials.encode(tokenizer, "hello <|endoftext|>");
        // --8<-- [end:specials-encode]
    }

    void specialsEncodeInto() {
        // --8<-- [start:specials-encode-into]
        Tokenizer tokenizer = null; // ...
        Specials specials = Specials.compile(tokenizer.vocabulary(), Set.of("<|endoftext|>"));
        IntSequence.Builder out = IntSequence.newBuilder();
        specials.encodeInto(tokenizer, "prefix <|endoftext|> suffix", out);
        IntSequence ids = out.build();
        // --8<-- [end:specials-encode-into]
    }

    void specialsNone() {
        // --8<-- [start:specials-none]
        Tokenizer tokenizer = null; // ...
        Specials none = Specials.none();
        IntSequence ids = none.encode(tokenizer, "just text");
        // --8<-- [end:specials-none]
    }

    void vocabularyIterate() {
        // --8<-- [start:vocabulary-iterate]
        Tokenizer tokenizer = null; // ...
        Vocabulary vocab = tokenizer.vocabulary();
        for (var entry : vocab) {
            String token = entry.getKey();
            int id = entry.getValue();
        }
        // --8<-- [end:vocabulary-iterate]
    }

    void byteLevel() {
        // --8<-- [start:bytelevel]
        String text = "Hello\n世界!";
        byte[] raw = text.getBytes(StandardCharsets.UTF_8);
        // raw bytes: 48 65 6c 6c 6f 0a e4 b8 96 e7 95 8c 21
        String symbols = ByteLevel.encode(raw);
        // symbols: "HelloĠä¸ŁçĥŃ!"
        byte[] decoded = ByteLevel.decode(symbols);
        String roundTripped = new String(decoded, StandardCharsets.UTF_8);
        // roundTripped: "Hello\n世界!"
        // --8<-- [end:bytelevel]
    }

    void modelCreate() {
        // --8<-- [start:model-create]
        String[] rankedTokensArray = {}; // ...
        Map<String, Integer> specialTokens = Map.of(); // ...
        Vocabulary vocab = Toknroll.vocabulary(specialTokens, rankedTokensArray);

        // Tiktoken BPE (OpenAI GPT family)
        List<MergeRule> mergeRules = List.of(); // ...
        TokenizationModel tiktokenModel = Toknroll.tiktokenModel(vocab, mergeRules);

        // SentencePiece BPE (Llama, Gemma, Mistral, etc.)
        TokenizationModel spModel = Toknroll.sentencePieceBpeModel(vocab, mergeRules);

        // SentencePiece with compact float scores
        float[] scores = {}; // ...
        TokenizationModel spScoreModel = Toknroll.sentencePieceBpeModel(vocab, scores);
        // --8<-- [end:model-create]
    }

    void intSequenceCreate() {
        // --8<-- [start:intsequence-create]
        IntSequence empty = IntSequence.empty();
        IntSequence literal = IntSequence.of(1, 2, 3);
        IntSequence wrapped = IntSequence.wrap(new int[] {1, 2, 3});
        IntSequence copied = IntSequence.copyOf(new int[] {1, 2, 3});
        IntSequence concat = literal.concat(wrapped);
        // --8<-- [end:intsequence-create]
    }
}
