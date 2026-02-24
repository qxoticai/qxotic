package com.qxotic.tokenizers.impl;

import com.qxotic.tokenizers.*;
import com.qxotic.tokenizers.advanced.Normalizer;
import com.qxotic.tokenizers.advanced.Splitter;
import com.qxotic.tokenizers.advanced.StandardTokenType;
import com.qxotic.tokenizers.advanced.SymbolCodec;
import java.io.ByteArrayOutputStream;
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

    public record MergeRank(int mergedTokenIndex, int rank) {}

    private final Map<IntPair, MergeRank> merges;
    private final SymbolCodec symbolCodec;
    private static final boolean useJavaReplacement = false;

    public GPT2Tokenizer(
            Vocabulary vocabulary,
            Normalizer normalizer,
            Splitter splitter,
            Map<IntPair, MergeRank> mergeRanks) {
        this(vocabulary, normalizer, splitter, mergeRanks, SymbolCodec.BYTE_LEVEL);
    }

    public GPT2Tokenizer(
            Vocabulary vocabulary,
            Normalizer normalizer,
            Splitter splitter,
            Map<IntPair, MergeRank> mergeRanks,
            SymbolCodec symbolCodec) {
        super(vocabulary, normalizer, splitter);
        this.merges = mergeRanks;
        this.symbolCodec = symbolCodec;
    }

    public GPT2Tokenizer(
            Vocabulary vocabulary,
            Normalizer normalizer,
            Splitter splitter,
            List<IntPair> mergeRanks) {
        this(vocabulary, normalizer, splitter, mergeRanks, SymbolCodec.BYTE_LEVEL);
    }

    public GPT2Tokenizer(
            Vocabulary vocabulary,
            Normalizer normalizer,
            Splitter splitter,
            List<IntPair> mergeRanks,
            SymbolCodec symbolCodec) {
        super(vocabulary, normalizer, splitter);
        this.merges = new HashMap<>(mergeRanks.size());
        this.symbolCodec = symbolCodec;
        for (int rank = 0; rank < mergeRanks.size(); rank++) {
            IntPair pair = mergeRanks.get(rank);
            int leftIndex = pair.left();
            int rightIndex = pair.right();
            String leftString = vocabulary.token(leftIndex);
            assert leftString != null;
            String rightString = vocabulary.token(rightIndex);
            assert rightString != null;
            int mergeIndex = vocabulary.id(leftString + rightString);
            this.merges.put(pair, new MergeRank(mergeIndex, rank));
        }
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
        IntSequence.Builder builder = IntSequence.newBuilder();
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
                builder.addAll(encode(part));
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

    private IntSequence encodeChunk(CharSequence chunk) {
        if (chunk.isEmpty()) {
            return IntSequence.empty();
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
            MergeRank bestMergeRank = null;
            int startPosition = 0;
            for (int i = 0; i + 1 < size; ++i) {
                IntPair candidatePair = new IntPair(tokens[i], tokens[i + 1]);
                MergeRank candidateMergeRank = merges.get(candidatePair);
                if (candidateMergeRank != null
                        && (bestMergeRank == null
                                || candidateMergeRank.rank() < bestMergeRank.rank())) {
                    bestMergeRank = candidateMergeRank;
                    startPosition = i;
                }
            }
            if (bestMergeRank == null) {
                // nothing else can be merged anymore
                break;
            }
            // otherwise let's merge the best pair (lowest rank)
            tokens[startPosition] = bestMergeRank.mergedTokenIndex();
            System.arraycopy(
                    tokens, startPosition + 2, tokens, startPosition + 1, size - startPosition - 2);
            --size;
        }
        if (tokens.length - size <= 8) {
            return IntSequence.wrap(tokens).subSequence(0, size);
        }
        return IntSequence.wrap(Arrays.copyOf(tokens, size));
    }

    @Override
    protected IntSequence encodeImpl(CharSequence text) {
        byte[] rawBytes =
                useJavaReplacement
                        ? text.toString().getBytes(StandardCharsets.UTF_8) // use ?
                        : getBytesWithReplacement(
                                text, REPLACEMENT_BYTES); // use \uFFFD (EF BF BD in UTF8)
        String encodedSymbols = symbolCodec.encodeBytes(rawBytes);
        return encodeChunk(encodedSymbols);
    }

    private static final byte[] REPLACEMENT_BYTES = {(byte) 0xEF, (byte) 0xBF, (byte) 0xBD};

    private static byte[] getBytesWithReplacement(CharSequence text, byte[] replacementBytes) {
        CharsetEncoder encoder =
                StandardCharsets.UTF_8
                        .newEncoder()
                        .onMalformedInput(CodingErrorAction.REPLACE)
                        .onUnmappableCharacter(CodingErrorAction.REPLACE)
                        .replaceWith(replacementBytes);
        ByteBuffer byteBuffer = null;
        try {
            byteBuffer = encoder.encode(CharBuffer.wrap(text));
        } catch (CharacterCodingException e) {
            // Should never happen with replacement.
            throw new RuntimeException(e);
        }
        byte[] bytes = new byte[byteBuffer.limit()];
        byteBuffer.get(bytes);
        return bytes;
    }

    @Override
    public byte[] decodeBytes(IntSequence tokens) {
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        for (int i = 0; i < tokens.length(); ++i) {
            int tokenIndex = tokens.intAt(i);
            String tokenString = vocabulary().token(tokenIndex);
            if (tokenString != null) {
                output.writeBytes(symbolCodec.decodeSymbols(tokenString));
            }
        }
        return output.toByteArray();
    }

    @Override
    public String toString() {
        return "GPT2 BPE Tokenizer";
    }
}
