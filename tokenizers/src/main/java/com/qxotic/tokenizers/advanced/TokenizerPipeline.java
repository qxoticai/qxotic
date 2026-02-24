package com.qxotic.tokenizers.advanced;

import com.qxotic.tokenizers.IntSequence;
import com.qxotic.tokenizers.Tokenizer;
import com.qxotic.tokenizers.Vocabulary;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Objects;

/**
 * A simplified tokenizer implementation that composes the core pipeline components: normalizer,
 * splitter, encoder, vocabulary, and decoder.
 *
 * <p>This class implements a minimal tokenization pipeline for mainstream inference:
 *
 * <ol>
 *   <li>Normalization - canonicalize the input text
 *   <li>Splitting - split into chunks
 *   <li>Encoding - convert chunks to IDs
 *   <li>Decoding - convert IDs back to text
 * </ol>
 *
 * <p>Example usage:
 *
 * <pre>
 * TokenizerPipeline pipeline = new TokenizerPipeline.Builder()
 *     .normalizer(Normalizer.unicode(Form.NFKC))
 *     .splitter(Splitter.identity())
 *     .encoder(chunk -> encodeChunk(chunk, vocab, merges))
 *     .vocabulary(vocab)
 *     .decoder(Decoder.CANONICAL)
 *     .build();
 *
 * IntSequence tokens = pipeline.encode("Hello, world!");
 * String decoded = pipeline.decode(tokens);
 * </pre>
 *
 * @see Tokenizer
 * @see Normalizer
 * @see Splitter
 * @see Encoder
 * @see Decoder
 */
public final class TokenizerPipeline implements Tokenizer {

    private final Normalizer normalizer;
    private final Splitter splitter;
    private final Encoder encoder;
    private final Vocabulary vocabulary;
    private final Decoder decoder;

    private TokenizerPipeline(Builder builder) {
        this.normalizer = Objects.requireNonNull(builder.normalizer, "normalizer is required");
        this.splitter = Objects.requireNonNull(builder.splitter, "splitter is required");
        this.encoder = Objects.requireNonNull(builder.encoder, "encoder is required");
        this.vocabulary = Objects.requireNonNull(builder.vocabulary, "vocabulary is required");
        this.decoder = Objects.requireNonNull(builder.decoder, "decoder is required");
    }

    @Override
    public IntSequence encode(String text) {
        // 1. Normalize
        CharSequence normalized = normalizer.apply(text);

        // 2. Split
        List<CharSequence> chunks = splitter.split(normalized);

        // 3. Encode chunks
        IntSequence.Builder builder = IntSequence.newBuilder();
        for (CharSequence chunk : chunks) {
            builder.addAll(encoder.encode(chunk));
        }
        return builder.build();
    }

    @Override
    public String decode(IntSequence tokens) {
        return decoder.decode(tokens, vocabulary);
    }

    @Override
    public byte[] decodeBytes(IntSequence tokens) {
        return decode(tokens).getBytes(StandardCharsets.UTF_8);
    }

    @Override
    public Vocabulary vocabulary() {
        return vocabulary;
    }

    /** Returns the normalizer used in this pipeline. */
    public Normalizer normalizer() {
        return normalizer;
    }

    /** Returns the splitter used in this pipeline. */
    public Splitter splitter() {
        return splitter;
    }

    /** Returns the encoder used in this pipeline. */
    public Encoder encoder() {
        return encoder;
    }

    /** Returns the decoder used in this pipeline. */
    public Decoder decoder() {
        return decoder;
    }

    /** Builder for creating TokenizerPipeline instances. */
    public static class Builder {
        private Normalizer normalizer = Normalizer.IDENTITY;
        private Splitter splitter = Splitter.identity();
        private Encoder encoder;
        private Vocabulary vocabulary;
        private Decoder decoder = Decoder.CANONICAL;

        /** Sets the normalizer. */
        public Builder normalizer(Normalizer normalizer) {
            this.normalizer = normalizer;
            return this;
        }

        /** Sets the splitter. */
        public Builder splitter(Splitter splitter) {
            this.splitter = splitter;
            return this;
        }

        /** Sets the encoder (required). */
        public Builder encoder(Encoder encoder) {
            this.encoder = encoder;
            return this;
        }

        /** Sets the vocabulary (required). */
        public Builder vocabulary(Vocabulary vocabulary) {
            this.vocabulary = vocabulary;
            return this;
        }

        /** Sets the decoder. */
        public Builder decoder(Decoder decoder) {
            this.decoder = decoder;
            return this;
        }

        /**
         * Builds the TokenizerPipeline.
         *
         * @return a new TokenizerPipeline
         * @throws NullPointerException if encoder or vocabulary is not set
         */
        public TokenizerPipeline build() {
            return new TokenizerPipeline(this);
        }
    }
}
