// Pure-Java phonemizer: an IVL2 lexicon of pre-phonemized words, plus a suffix fallback.
// Unknown words fall through to espeak-ng per run when installed, else letter-to-sound rules.
// Read from a file the caller names, or from the classpath when a jar or image bundles one.
package com.qxotic.jinfer.models.inflect2.frontend;

import com.qxotic.jinfer.models.inflect2.Symbols;
import java.io.IOException;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

final class LexiconPhonemizer implements Phonemizer {

    private static final String RESOURCE = "/lexicon.bin";

    /** Word separator in the symbol table - the model hears a space as a symbol of its own. */
    private static final int SPACE = Symbols.idOf(' ');

    /** IPA for one punctuation-free run; the fallback channel, espeak-ng's {@code ipaRun}. */
    interface RunFallback {
        String ipa(String run) throws IOException;
    }

    private final Map<String, int[]> lexicon;
    private final RunFallback fallback; // espeak-ng, or null: letter-to-sound guesses instead
    private boolean warned;

    private LexiconPhonemizer(Map<String, int[]> lexicon) {
        this(lexicon, null);
    }

    private LexiconPhonemizer(Map<String, int[]> lexicon, RunFallback fallback) {
        this.lexicon = lexicon;
        this.fallback = fallback;
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

    @Override
    public Phonemizer withEspeakFallback() {
        EspeakPhonemizer espeak = EspeakPhonemizer.tryCreate();
        return espeak == null ? this : new LexiconPhonemizer(lexicon, espeak::ipaRun);
    }

    /** A small in-memory lexicon, for tests. */
    static LexiconPhonemizer of(Map<String, int[]> lexicon, RunFallback fallback) {
        return new LexiconPhonemizer(lexicon, fallback);
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
     * <p>Words the lexicon does not know are never silent. With an espeak fallback, a whole
     * punctuation-free run containing an unknown word goes to espeak - stress is contextual, so
     * per-word fallback would stamp a primary stress on every function word (see {@link
     * EspeakPhonemizer}). Without one, each unknown word is guessed by {@link LetterToSound} and
     * the guess count is logged once, because a wrong pronunciation should be audible work, not an
     * invisible gap.
     */
    @Override
    public int[] phonemize(String text) throws IOException {
        Emitter out = new Emitter();
        var runWords = new ArrayList<String>();
        var runIds = new ArrayList<int[]>();
        boolean runHasUnknown = false;
        for (String token : text.split("\\s+")) {
            if (token.isEmpty()) continue;
            int start = EspeakPhonemizer.wordStart(token);
            int end = EspeakPhonemizer.wordEnd(token);
            if (start > 0) out.put(Symbols.toRaw(token.substring(0, start))); // an opening quote
            if (end > start) {
                // Keep the spelling the writer used. Only the lexicon key is lowercased: espeak
                // reads capitals as information (it says "GraalVM" as "graal vee em" and
                // "graalvm" as one mangled word), and letter-to-sound lowercases for itself.
                String word = token.substring(start, end);
                int[] ids = lookup(word.toLowerCase(Locale.ROOT));
                if (ids == null || ids.length == 0) runHasUnknown = true;
                runWords.add(word);
                runIds.add(ids);
            }
            if (end < token.length()) { // trailing punctuation closes the run
                out.put(runSymbols(runWords, runIds, runHasUnknown));
                out.put(Symbols.toRaw(token.substring(end)));
                runWords.clear();
                runIds.clear();
                runHasUnknown = false;
            }
        }
        out.put(runSymbols(runWords, runIds, runHasUnknown));
        if (guessedCount > 0 && !warned) {
            warned = true;
            System.getLogger("jinfer.inflect2")
                    .log(
                            System.Logger.Level.INFO,
                            "{0} word(s) are not in the lexicon; pronounced by letter-to-sound"
                                    + " rules. Correct one with a word override, or install"
                                    + " espeak-ng to cover them all",
                            guessedCount);
        }
        return Symbols.blankIntersperse(out.symbols());
    }

    private int guessedCount;

    /**
     * The buffered punctuation-free run as raw symbol ids: espeak when it covers the unknowns,
     * lexicon plus letter-to-sound guesses otherwise.
     */
    private int[] runSymbols(List<String> words, List<int[]> ids, boolean hasUnknown)
            throws IOException {
        if (words.isEmpty()) return new int[0];
        if (hasUnknown && fallback != null)
            return Symbols.toRaw(fallback.ipa(String.join(" ", words)));
        int total = words.size() - 1; // spaces between words
        for (int i = 0; i < ids.size(); i++) {
            int[] word = ids.get(i);
            if (word == null || word.length == 0) {
                word = Symbols.toRaw(LetterToSound.guess(words.get(i)));
                guessedCount++;
                ids.set(i, word);
            }
            total += word.length;
        }
        int[] out = new int[total];
        int at = 0;
        for (int i = 0; i < ids.size(); i++) {
            if (i > 0) out[at++] = SPACE;
            int[] word = ids.get(i);
            System.arraycopy(word, 0, out, at, word.length);
            at += word.length;
        }
        return out;
    }

    /** Space-separated symbol stream assembly: every block lands behind one separator. */
    private static final class Emitter {
        private int[] symbols = new int[64];
        private int length;

        void put(int[] block) {
            if (block.length == 0) return;
            int needed = length + block.length + (length > 0 ? 1 : 0);
            if (needed > symbols.length) symbols = Arrays.copyOf(symbols, needed * 2);
            if (length > 0) symbols[length++] = SPACE;
            System.arraycopy(block, 0, symbols, length, block.length);
            length += block.length;
        }

        int[] symbols() {
            return Arrays.copyOf(symbols, length);
        }
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
        try {
            ByteBuffer buffer = ByteBuffer.wrap(data).order(ByteOrder.LITTLE_ENDIAN);
            byte[] magic = new byte[MAGIC.length()];
            buffer.get(magic);
            if (!MAGIC.equals(new String(magic, StandardCharsets.US_ASCII)))
                throw new IOException("not a lexicon: bad magic");
            int version = buffer.getShort();
            if (version != VERSION) throw new IOException("unsupported lexicon version " + version);
            buffer.position(buffer.position() + HEADER_PAD);
            int count = buffer.getInt();
            if (count < 0) throw new IOException("negative lexicon entry count");
            buffer.position(buffer.position() + 8);
            long entriesAt = buffer.getLong();
            if (entriesAt < 0 || entriesAt > data.length)
                throw new IOException("invalid lexicon entry offset " + entriesAt);
            // shared + fresh + symbols: the smallest entry is three bytes
            if (count > (data.length - entriesAt) / 3)
                throw new IOException("lexicon entry count exceeds the file size");

            Map<String, int[]> lexicon = HashMap.newHashMap(count);
            buffer.position((int) entriesAt);
            byte[] word = new byte[MAX_WORD];
            int length = 0;
            for (int entry = 0; entry < count; entry++) {
                int shared = Byte.toUnsignedInt(buffer.get());
                int fresh = Byte.toUnsignedInt(buffer.get());
                if (shared > length)
                    throw new IOException("lexicon entry " + entry + " out of order");
                length = shared + fresh;
                if (length > word.length) word = Arrays.copyOf(word, length * 2);
                buffer.get(word, shared, fresh);
                int symbols = Byte.toUnsignedInt(buffer.get());
                if (symbols == 0) continue;
                int[] ids = new int[symbols];
                for (int i = 0; i < symbols; i++) {
                    ids[i] = Byte.toUnsignedInt(buffer.get());
                    if (ids[i] >= Symbols.count())
                        throw new IOException(
                                "lexicon entry " + entry + " has invalid symbol " + ids[i]);
                }
                lexicon.put(new String(word, 0, length, StandardCharsets.US_ASCII), ids);
            }
            return lexicon;
        } catch (BufferUnderflowException | IllegalArgumentException e) {
            throw new IOException("truncated or malformed lexicon", e);
        }
    }
}
