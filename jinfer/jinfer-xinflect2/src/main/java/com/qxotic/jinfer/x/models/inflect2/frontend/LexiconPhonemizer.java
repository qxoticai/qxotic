// Pure-Java phonemizer: an IVL2 lexicon of pre-phonemized words, plus a suffix fallback.
// Read from a file the caller names, or from the classpath when a jar or image bundles one.
package com.qxotic.jinfer.x.models.inflect2.frontend;

import com.qxotic.jinfer.x.models.inflect2.Symbols;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

final class LexiconPhonemizer implements Phonemizer {

    private static final String RESOURCE = "/lexicon.bin";

    /** Word separator in the symbol table - the model hears a space as a symbol of its own. */
    private static final int SPACE = Symbols.idOf(' ');

    private final Map<String, int[]> lexicon;
    private boolean warned;

    private LexiconPhonemizer(Map<String, int[]> lexicon) {
        this.lexicon = lexicon;
    }

    /**
     * The lexicon at {@code path}. THROWS when it cannot be read or parsed: a caller who named a
     * file and silently got a fallback has been lied to, and spelling every word out letter by
     * letter is a far worse outcome than a load failure.
     */
    static LexiconPhonemizer read(Path path) throws IOException {
        return new LexiconPhonemizer(readTrie(Files.readAllBytes(path)));
    }

    /**
     * The lexicon bundled on the classpath, or null when the jar or image carries none. This is the
     * rung a single-file native binary lands on: it has no directory to sit beside.
     */
    static LexiconPhonemizer bundled() throws IOException {
        try (var in = LexiconPhonemizer.class.getResourceAsStream(RESOURCE)) {
            return in == null ? null : new LexiconPhonemizer(readTrie(in.readAllBytes()));
        }
    }

    /**
     * Look each word up and join with the separator symbol. The text is expected to be normalized
     * already (see {@link Phonemizer}).
     *
     * <p>Trailing punctuation is split off before the lookup and re-emitted after it, exactly as
     * the espeak path does - it carries the prosody the model needs, and a lexicon keyed on bare
     * words would otherwise miss every clause-final word in ordinary prose ("grows." and "thin."
     * are in the lexicon; "grows." and "thin." with the stop attached are not).
     *
     * <p>There is no letter-to-sound model behind this one, so a word it does not know is dropped -
     * and says so once on stderr, because silence for a missing word is invisible downstream.
     */
    @Override
    public int[] phonemize(String text) {
        int[] symbols = new int[64];
        int length = 0;
        int dropped = 0;
        for (String token : text.split("\\s+")) {
            if (token.isEmpty()) continue;
            int end = EspeakPhonemizer.wordEnd(token);
            int[] ids = end > 0 ? lookup(token.substring(0, end).toLowerCase()) : null;
            if (end > 0 && (ids == null || ids.length == 0)) dropped++;
            int[] marks = Symbols.toRaw(token.substring(end));
            int width = (ids == null ? 0 : ids.length) + marks.length;
            if (width == 0) continue;
            if (length + width + 1 > symbols.length)
                symbols = Arrays.copyOf(symbols, (length + width + 1) * 2);
            if (length > 0) symbols[length++] = SPACE;
            if (ids != null) {
                System.arraycopy(ids, 0, symbols, length, ids.length);
                length += ids.length;
            }
            System.arraycopy(marks, 0, symbols, length, marks.length);
            length += marks.length;
        }
        if (dropped > 0 && !warned) {
            warned = true;
            System.getLogger("jinfer.inflect2")
                    .log(
                            System.Logger.Level.WARNING,
                            "{0} word(s) are not in the lexicon and were left unspoken; install"
                                    + " espeak-ng and remove the lexicon to phonemize them"
                                    + " instead",
                            dropped);
        }
        return Symbols.blankIntersperse(Arrays.copyOf(symbols, length));
    }

    /** Exact match, else a known suffix split off a stem that is in the lexicon. */
    private int[] lookup(String word) {
        int[] exact = lexicon.get(word);
        if (exact != null) return exact;
        for (Suffix suffix : SUFFIXES) {
            if (!word.endsWith(suffix.ending()) || word.length() <= suffix.ending().length() + 1)
                continue;
            String stem =
                    word.substring(0, word.length() - suffix.ending().length()) + suffix.stemTail();
            int[] stemIds = lexicon.get(stem);
            int[] suffixIds = lexicon.get(suffix.spoken());
            if (stemIds == null || suffixIds == null) continue;
            int[] joined = Arrays.copyOf(stemIds, stemIds.length + suffixIds.length);
            System.arraycopy(suffixIds, 0, joined, stemIds.length, suffixIds.length);
            return joined;
        }
        return null;
    }

    /** An inflection: strip {@code ending}, restore {@code stemTail}, then say {@code spoken}. */
    private record Suffix(String ending, String stemTail, String spoken) {}

    private static final Suffix[] SUFFIXES = {
        new Suffix("ies", "y", "z"),
        new Suffix("'s", "", "z"),
        new Suffix("s'", "", "z"),
        new Suffix("es", "", "z"),
        new Suffix("s", "", "z"),
        new Suffix("ied", "y", "d"),
        new Suffix("ed", "", "d"),
        new Suffix("ing", "", "ng"),
    };

    // ── IVL2 lexicon format ───────────────────────────────────────────────

    private static final String MAGIC = "IVL2";
    private static final int VERSION = 1;
    private static final int HEADER_PAD = 2 + 32; // flags + reserved, after magic and version
    private static final int MAX_WORD = 64;

    /**
     * Words are delta-coded against their predecessor: shared-prefix length, the bytes that differ,
     * then the word's symbol ids. Sorted order is what makes the prefix reuse work.
     */
    private static Map<String, int[]> readTrie(byte[] data) throws IOException {
        ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
        byte[] magic = new byte[MAGIC.length()];
        buffer.get(magic);
        if (!MAGIC.equals(new String(magic, StandardCharsets.US_ASCII)))
            throw new IOException("not a lexicon: bad magic");
        int version = buffer.getShort();
        if (version != VERSION) throw new IOException("unsupported lexicon version " + version);
        buffer.position(buffer.position() + HEADER_PAD);
        int count = buffer.getInt();
        buffer.position(buffer.position() + 8);
        int entriesAt = (int) buffer.getLong();

        Map<String, int[]> lexicon = HashMap.newHashMap(count);
        buffer.position(entriesAt);
        byte[] word = new byte[MAX_WORD];
        int length = 0;
        for (int entry = 0; entry < count && buffer.hasRemaining(); entry++) {
            int shared = Byte.toUnsignedInt(buffer.get());
            int fresh = Byte.toUnsignedInt(buffer.get());
            if (shared > length) throw new IOException("lexicon entry " + entry + " out of order");
            length = shared + fresh;
            if (length > word.length) word = Arrays.copyOf(word, length * 2);
            buffer.get(word, shared, fresh);
            int symbols = Byte.toUnsignedInt(buffer.get());
            if (symbols == 0) continue;
            int[] ids = new int[symbols];
            for (int i = 0; i < symbols; i++) ids[i] = Byte.toUnsignedInt(buffer.get());
            lexicon.put(new String(word, 0, length, StandardCharsets.US_ASCII), ids);
        }
        return lexicon;
    }
}
