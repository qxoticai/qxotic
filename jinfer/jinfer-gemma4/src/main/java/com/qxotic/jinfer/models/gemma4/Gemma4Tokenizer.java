package com.qxotic.jinfer.models.gemma4;

import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.Vocabulary;
import java.nio.ByteBuffer;
import java.nio.CharBuffer;
import java.util.Arrays;
import java.util.BitSet;

/**
 * Gemma 4's tokenizer, fast: the generic one, fed words instead of lines, and a word cache.
 *
 * <p>The generic tokenizer merges each line whole. Every merge builds a vocabulary token, so a line
 * can be cut wherever no token spans the cut, and the words merge exactly as the line would. A cut
 * goes before a space, unless some token holds the char before it followed by a space (Gemma 4's
 * one such char is {@code >}, for its token {@code > </}).
 */
final class Gemma4Tokenizer implements Tokenizer {

    private static final int CACHE_SLOTS = 1 << 20;
    private static final int CACHED_WORD = 64; // chars: longer words skip the cache, bounding it
    private static final long FNV_OFFSET = 0xCBF29CE484222325L; // a word's hash: FNV-1a
    private static final long FNV_PRIME = 0x100000001B3L;

    private final Tokenizer generic;
    private final BitSet glue = new BitSet(); // chars a space is never cut from

    /**
     * Words to ids, one slot per hash, shared by every thread without a lock: an entry is
     * immutable, so a thread that sees one sees it whole, and one lost to a race or overwritten by
     * another word is only a miss.
     */
    private final Entry[] cache = new Entry[CACHE_SLOTS];

    private record Entry(char[] word, int[] ids) {}

    Gemma4Tokenizer(Tokenizer generic) {
        this.generic = generic;
        Vocabulary vocabulary = generic.vocabulary();
        for (int id = 0; id < vocabulary.size(); id++) {
            String token = vocabulary.token(id);
            for (int i = 1; token != null && i < token.length(); i++) {
                if (token.charAt(i) == ' ' && token.charAt(i - 1) != ' ') {
                    glue.set(token.charAt(i - 1));
                }
            }
        }
    }

    @Override
    public void encodeInto(CharSequence text, int start, int end, IntSequence.Builder out) {
        // a literal U+2581 is a space, as the generic tokenizer normalizes it
        char[] chars = text.subSequence(start, end).toString().replace('\u2581', ' ').toCharArray();
        // lines and newline runs, as the generic tokenizer splits, then words; each hashed here
        long hash = FNV_OFFSET;
        int from = 0;
        for (int i = 0; i < chars.length; i++) {
            char c = chars[i];
            if (i > from) {
                char before = chars[i - 1];
                if ((c == '\n') != (before == '\n')
                        || (c == ' ' && before != ' ' && !glue.get(before))) {
                    word(chars, from, i, hash, out);
                    from = i;
                    hash = FNV_OFFSET;
                }
            }
            hash = (hash ^ c) * FNV_PRIME;
        }
        if (from < chars.length) word(chars, from, chars.length, hash, out);
    }

    private void word(char[] chars, int from, int to, long hash, IntSequence.Builder out) {
        if (to - from > CACHED_WORD) {
            generic.encodeInto(CharBuffer.wrap(chars), from, to, out);
            return;
        }
        int slot = (int) (hash ^ (hash >>> 32)) & (CACHE_SLOTS - 1);
        Entry entry = cache[slot];
        if (entry == null || !Arrays.equals(entry.word, 0, entry.word.length, chars, from, to)) {
            int[] ids = generic.encodeToArray(CharBuffer.wrap(chars, from, to - from));
            cache[slot] = entry = new Entry(Arrays.copyOfRange(chars, from, to), ids);
        }
        for (int id : entry.ids) out.add(id);
    }

    @Override
    public int countTokens(CharSequence text, int start, int end) {
        return encode(text.subSequence(start, end)).length();
    }

    @Override
    public Vocabulary vocabulary() {
        return generic.vocabulary();
    }

    @Override
    public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
        return generic.decodeBytesInto(tokens, tokenStartIndex, out);
    }
}
