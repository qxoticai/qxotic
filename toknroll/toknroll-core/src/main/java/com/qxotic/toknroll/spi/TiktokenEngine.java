package com.qxotic.toknroll.spi;

/**
 * A fast encoder for the tiktoken family: a byte-level BPE model behind one of the pre-tokenization
 * regexes below. Found through {@link java.util.ServiceLoader}; toknroll-vector provides one.
 *
 * <p>Core asks for an encoder the first time a tokenizer of the family encodes, checks it against
 * its own encoder on a sample that covers every kind of piece, and uses it from then on. An engine
 * that declines, disagrees or fails is passed over, and core encodes by itself. Setting the system
 * property {@code toknroll.engine} to {@code none} passes over every engine for the models whose
 * first encode comes after it.
 */
public interface TiktokenEngine {

    /**
     * The regex of cl100k and Llama 3: {@code
     * (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+| \p{N}{1,3}|
     * ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+}.
     */
    int CL100K = 0;

    /** {@link #CL100K} with the contraction spelled {@code '[sS]|'[tT]|...}: no long s. */
    int CL100K_ASCII = 1;

    /** {@link #CL100K} with one digit at a time, {@code \p{N}}: Qwen2 and its descendants. */
    int QWEN2 = 2;

    /** {@link #QWEN2} with the contraction spelled {@code '[sS]|'[tT]|...}. */
    int QWEN2_ASCII = 3;

    /**
     * The regex of r50k and GPT-2's byte level pre-tokenizer: {@code 's|'t|'re|'ve|'m|'ll|'d|
     * ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+}.
     */
    int R50K = 4;

    /**
     * The regex of o200k, gpt-4o and gpt-oss: {@code [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}
     * \p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|} the same with {@code +} and
     * {@code *} swapped {@code |\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+}.
     */
    int O200K = 5;

    /** {@link #O200K} without the contraction suffix and with one digit at a time: Tekken. */
    int TEKKEN = 6;

    /**
     * An encoder for {@code model} under the regex {@code rules}, or null if this engine has none.
     */
    Encoder encoder(TiktokenTables model, int rules);

    /** Encodes whole texts, from any number of threads at once. */
    interface Encoder {
        /**
         * Appends the ids of {@code text[start, end)} to {@code out}. A surrogate without its
         * partner encodes as U+FFFD.
         */
        void encode(CharSequence text, int start, int end, IdBuffer out);

        /** Appends the ids of UTF-8 text; ill-formed bytes are taken as they are. */
        void encodeUtf8(byte[] utf8, int start, int end, IdBuffer out);
    }
}
