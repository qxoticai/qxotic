package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.*;
import com.qxotic.toknroll.advanced.Normalizer;
import com.qxotic.toknroll.advanced.Splitter;
import com.qxotic.toknroll.advanced.StandardTokenType;
import com.qxotic.toknroll.advanced.SymbolCodec;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.nio.charset.CharacterCodingException;
import java.nio.charset.CharsetEncoder;
import java.nio.charset.CodingErrorAction;
import java.nio.charset.StandardCharsets;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/** Example implementation of a BPE tokenizer for GPT2. */
public class GPT2Tokenizer extends AbstractTokenizer {

    private final LongLongMap merges;
    private final SymbolCodec symbolCodec;
    private final byte[][] tokenBytesById;

    public GPT2Tokenizer(
            Vocabulary vocabulary,
            Normalizer normalizer,
            Splitter splitter,
            LongLongMap mergeRanks) {
        this(vocabulary, normalizer, splitter, mergeRanks, SymbolCodec.BYTE_LEVEL);
    }

    public GPT2Tokenizer(
            Vocabulary vocabulary,
            Normalizer normalizer,
            Splitter splitter,
            LongLongMap mergeRanks,
            SymbolCodec symbolCodec) {
        super(vocabulary, normalizer, splitter);
        this.merges = mergeRanks;
        this.symbolCodec = symbolCodec;
        this.tokenBytesById = buildTokenBytesById(vocabulary, symbolCodec);
    }

    public GPT2Tokenizer(
            Vocabulary vocabulary, Normalizer normalizer, Splitter splitter, List<?> mergeRanks) {
        this(vocabulary, normalizer, splitter, mergeRanks, SymbolCodec.BYTE_LEVEL);
    }

    public GPT2Tokenizer(
            Vocabulary vocabulary,
            Normalizer normalizer,
            Splitter splitter,
            List<?> mergeRanks,
            SymbolCodec symbolCodec) {
        super(vocabulary, normalizer, splitter);
        this.symbolCodec = symbolCodec;
        long[] keys = new long[mergeRanks.size()];
        long[] values = new long[mergeRanks.size()];
        for (int rank = 0; rank < mergeRanks.size(); rank++) {
            long pair = toPair(mergeRanks.get(rank));
            int leftIndex = IntPair.left(pair);
            int rightIndex = IntPair.right(pair);
            String leftString = vocabulary.token(leftIndex);
            assert leftString != null;
            String rightString = vocabulary.token(rightIndex);
            assert rightString != null;
            int mergeIndex = vocabulary.id(leftString + rightString);
            keys[rank] = pair;
            values[rank] = IntPair.of(mergeIndex, rank);
        }
        this.merges = new LongLongMap(keys, values);
        this.tokenBytesById = buildTokenBytesById(vocabulary, symbolCodec);
    }

    private static long toPair(Object value) {
        if (value instanceof long[]) {
            long[] arr = (long[]) value;
            if (arr.length == 0) {
                throw new IllegalArgumentException("merge pair array cannot be empty");
            }
            return arr[0];
        }
        if (value instanceof IntPair) {
            IntPair pair = (IntPair) value;
            return IntPair.of(pair.first(), pair.second());
        }
        throw new IllegalArgumentException(
                "Unsupported merge pair type: " + (value == null ? "null" : value.getClass()));
    }

    /**
     * Unlike {@link #encodeOrdinary(String)}, this function handles special tokens.
     * allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens if
     * none_raise, then an error is raised if any special token is encountered in text this is the
     * default tiktoken behavior right now as well any other behavior is either annoying, or a major
     * footgun.
     */
    private IntSequence encode(String text, Set<String> allowedSpecial) {
        // decode the user desire w.r.t. handling of special tokens
        Set<String> special = allowedSpecial;

        // Should be valid tokens.
        assert allowedSpecial.stream().allMatch(vocabulary::contains);

        // Allowed tokens should be all special.
        assert allowedSpecial.stream()
                .allMatch(
                        tokenString -> {
                            int tokenIndex = vocabulary.id(tokenString);
                            return !vocabulary()
                                    .isTokenOfType(tokenIndex, StandardTokenType.NORMAL);
                        });

        if (special.isEmpty()) {
            // shortcut: if no special tokens, just use the ordinary encoding
            return encode(text);
        }

        // otherwise, we have to be careful with potential special tokens in text
        // we handle special tokens by splitting the text
        // based on the occurrence of any exact match with any of the special tokens
        IntSequence.Builder builder = IntSequence.newBuilder(Math.max(8, text.length()));
        for (String part : splitWithSeparators(text, allowedSpecial)) {
            // now all the special characters are separated from the rest of the text
            // all chunks of text are encoded separately, then results are joined
            if (special.contains(part)) {
                // this is a special token, encode it separately as a special case
                int tokenIndex = vocabulary.id(part);
                assert tokenIndex >= 0;
                builder.add(tokenIndex);
            } else {
                // this is an ordinary sequence, encode it normally
                encodeInto(part, builder);
            }
        }
        return builder.build();
    }

    private static List<String> splitWithSeparators(String text, Set<String> separators) {
        String pattern =
                separators.stream().map(Pattern::quote).collect(Collectors.joining("|", "(", ")"));

        Matcher matcher = Pattern.compile(pattern).matcher(text);
        List<String> parts = new ArrayList<>();
        int lastEnd = 0;
        while (matcher.find()) {
            int start = matcher.start();
            if (start > lastEnd) {
                parts.add(text.substring(lastEnd, start));
            }
            parts.add(matcher.group());
            lastEnd = matcher.end();
        }
        if (text.length() > lastEnd) {
            parts.add(text.substring(lastEnd));
        }
        return parts;
    }

    private void encodeChunkInto(CharSequence chunk, IntSequence.Builder out) {
        if (chunk.length() == 0) {
            return;
        }

        int[] tokens = new int[chunk.length()];
        int size = 0;

        // return the token ids
        // let's begin. first, convert all bytes to integers in range 0..255
        for (int i = 0; i < chunk.length(); i++) {
            char ch = chunk.charAt(i);
            int tokenIndex = this.vocabulary.id(String.valueOf(ch));
            assert tokenIndex >= 0;
            tokens[size++] = tokenIndex;
        }

        while (size >= 2) {
            // find the pair with the lowest merge rank
            long bestValue = IntPair.NONE;
            int bestRank = Integer.MAX_VALUE;
            int startPosition = 0;
            for (int i = 0; i + 1 < size; ++i) {
                long candidateKey = IntPair.of(tokens[i], tokens[i + 1]);
                long candidateValue = merges.get(candidateKey);
                if (candidateValue != IntPair.NONE) {
                    int candidateRank = IntPair.right(candidateValue);
                    if (candidateRank < bestRank) {
                        bestValue = candidateValue;
                        bestRank = candidateRank;
                        startPosition = i;
                    }
                }
            }
            if (bestValue == IntPair.NONE) {
                // nothing else can be merged anymore
                break;
            }
            // otherwise let's merge the best pair (lowest rank)
            tokens[startPosition] = IntPair.left(bestValue);
            System.arraycopy(
                    tokens, startPosition + 2, tokens, startPosition + 1, size - startPosition - 2);
            --size;
        }
        out.ensureCapacity(out.size() + size);
        for (int i = 0; i < size; i++) {
            out.add(tokens[i]);
        }
    }

    @Override
    protected IntSequence encodeImpl(CharSequence text) {
        IntSequence.Builder out = IntSequence.newBuilder(Math.max(8, text.length()));
        encodeImplInto(text, out);
        return out.build();
    }

    @Override
    public int countTokens(CharSequence text) {
        return encode(Objects.requireNonNull(text, "text")).length();
    }

    @Override
    protected void encodeImplInto(CharSequence text, IntSequence.Builder out) {
        // Use U+FFFD (EF BF BD in UTF-8) for malformed surrogates, matching Python reference
        // implementations (tiktoken, HuggingFace). Note: String.getBytes(UTF_8) would use '?'
        // (0x3F) instead, which produces different token IDs.
        byte[] rawBytes = getBytesWithReplacement(text, REPLACEMENT_BYTES);
        String encodedSymbols = symbolCodec.encodeBytes(rawBytes);
        encodeChunkInto(encodedSymbols, out);
    }

    private static final byte[] REPLACEMENT_BYTES = {(byte) 0xEF, (byte) 0xBF, (byte) 0xBD};

    private static byte[] getBytesWithReplacement(CharSequence text, byte[] replacementBytes) {
        CharsetEncoder encoder =
                StandardCharsets.UTF_8
                        .newEncoder()
                        .onMalformedInput(CodingErrorAction.REPLACE)
                        .onUnmappableCharacter(CodingErrorAction.REPLACE)
                        .replaceWith(replacementBytes);
        try {
            ByteBuffer byteBuffer = encoder.encode(CharBuffer.wrap(text));
            byte[] bytes = new byte[byteBuffer.limit()];
            byteBuffer.get(bytes);
            return bytes;
        } catch (CharacterCodingException e) {
            // Should never happen with replacement.
            throw new RuntimeException(e);
        }
    }

    @Override
    public byte[] decodeBytes(IntSequence tokens) {
        int totalBytes = decodedByteLength(tokens);
        int length = tokens.length();

        byte[] out = new byte[totalBytes];
        int offset = 0;
        for (int i = 0; i < length; i++) {
            byte[] chunk = tokenBytes(tokens.intAt(i));
            System.arraycopy(chunk, 0, out, offset, chunk.length);
            offset += chunk.length;
        }
        return out;
    }

    @Override
    public int countBytes(IntSequence tokens) {
        return decodedByteLength(tokens);
    }

    @Override
    public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
        Objects.requireNonNull(tokens, "tokens");
        Objects.requireNonNull(out, "out");
        int length = tokens.length();
        if (tokenStartIndex < 0 || tokenStartIndex > length) {
            throw new IndexOutOfBoundsException("tokenStartIndex: " + tokenStartIndex);
        }
        if (tokenStartIndex == length) {
            return 0;
        }

        int consumed = 0;
        for (int i = tokenStartIndex; i < length; i++) {
            byte[] chunk = tokenBytes(tokens.intAt(i));
            if (chunk.length > out.remaining()) {
                if (consumed == 0) {
                    throw insufficientSpace(i, chunk.length, out.remaining());
                }
                break;
            }
            out.put(chunk);
            consumed++;
        }
        return consumed;
    }

    private int decodedByteLength(IntSequence tokens) {
        Objects.requireNonNull(tokens, "tokens");
        int length = tokens.length();
        int totalBytes = 0;
        for (int i = 0; i < length; i++) {
            totalBytes += tokenBytes(tokens.intAt(i)).length;
        }
        return totalBytes;
    }

    private byte[] tokenBytes(int tokenId) {
        if (tokenId < 0 || tokenId >= tokenBytesById.length) {
            throw new NoSuchElementException(String.valueOf(tokenId));
        }
        byte[] bytes = tokenBytesById[tokenId];
        if (bytes == null) {
            throw new NoSuchElementException(String.valueOf(tokenId));
        }
        return bytes;
    }

    private static byte[][] buildTokenBytesById(Vocabulary vocabulary, SymbolCodec symbolCodec) {
        int maxId = -1;
        for (Map.Entry<String, Integer> entry : vocabulary) {
            if (entry.getValue() > maxId) {
                maxId = entry.getValue();
            }
        }
        byte[][] table = new byte[Math.max(0, maxId + 1)][];
        for (Map.Entry<String, Integer> entry : vocabulary) {
            int tokenId = entry.getValue();
            table[tokenId] = symbolCodec.decodeSymbols(entry.getKey());
        }
        return table;
    }

    private static IllegalArgumentException insufficientSpace(
            int tokenIndex, int needed, int remaining) {
        return new IllegalArgumentException(
                "Not enough output space for token at index "
                        + tokenIndex
                        + ": need "
                        + needed
                        + ", remaining "
                        + remaining);
    }

    @Override
    public String toString() {
        return "GPT2 BPE Tokenizer";
    }
}
