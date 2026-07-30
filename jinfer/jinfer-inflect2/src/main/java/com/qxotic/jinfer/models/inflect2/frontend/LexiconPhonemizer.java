// Pure-Java phonemizer: an IVL2 lexicon of pre-phonemized words, plus a suffix fallback.
// The lexicon ships as a classpath resource; -Dinflect.lexicon=<path> overrides it.
package com.qxotic.jinfer.models.inflect2.frontend;

import com.qxotic.jinfer.models.inflect2.Symbols;
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
    private static final String OVERRIDE_PROPERTY = "inflect.lexicon";

    /** Word separator in the symbol table — the model hears a space as a symbol of its own. */
    private static final int SPACE = Symbols.idOf(' ');

    private final Map<String, int[]> lexicon;

    private LexiconPhonemizer(Map<String, int[]> lexicon) {
        this.lexicon = lexicon;
    }

    /**
     * The bundled lexicon, or the {@code -Dinflect.lexicon} override, or null when there is none. A
     * lexicon that is present but unreadable says so — silently spelling every word out letter by
     * letter is worse than one line on stderr.
     */
    static LexiconPhonemizer tryCreate() {
        String override = System.getProperty(OVERRIDE_PROPERTY);
        try {
            if (override != null)
                return new LexiconPhonemizer(readTrie(Files.readAllBytes(Path.of(override))));
            try (var in = LexiconPhonemizer.class.getResourceAsStream(RESOURCE)) {
                return in == null ? null : new LexiconPhonemizer(readTrie(in.readAllBytes()));
            }
        } catch (IOException e) {
            System.err.println("[inflect2] lexicon unusable (" + e.getMessage() + ")");
            return null;
        }
    }

    /**
     * Look each word up and join with the separator symbol. The text is expected to be normalized
     * already (see {@link Phonemizer}); a word that is not in the lexicon is dropped, since there
     * is no letter-to-sound model behind this one.
     */
    @Override
    public int[] phonemize(String text) {
        int[] symbols = new int[64];
        int length = 0;
        for (String word : text.split("\\s+")) {
            if (word.isEmpty()) continue;
            int[] ids = lookup(word.toLowerCase());
            if (ids == null || ids.length == 0) continue;
            if (length + ids.length + 1 > symbols.length)
                symbols = Arrays.copyOf(symbols, (length + ids.length + 1) * 2);
            if (length > 0) symbols[length++] = SPACE;
            System.arraycopy(ids, 0, symbols, length, ids.length);
            length += ids.length;
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
