// Pure-Java grapheme-to-phoneme: an IVL2 lexicon of pre-phonemized words plus a suffix fallback.
// A run with an unknown word goes to espeak-ng whole when it is installed, else the unknown word
// is guessed by letter-to-sound rules. Read from a file the caller names, or from the classpath
// when a jar or image bundles one.
package com.qxotic.jinfer.models.inflect2.frontend;

import com.qxotic.jinfer.models.inflect2.Inflect2;
import java.io.IOException;
import java.nio.BufferUnderflowException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.UnaryOperator;

/**
 * The lexicon is espeak-ng's own output, memoised: every entry is the IPA espeak would produce for
 * that word, so the lexicon and the espeak fallback are one front end at two speeds, not two
 * dialects. {@link #ipa} is the grapheme-to-phoneme step of a {@link com.qxotic.jinfer.Phonemizer};
 * it sees one punctuation-free run at a time.
 */
public final class Lexicon {

    private static final String RESOURCE = "/lexicon.bin";

    private final Map<String, String> words;
    private final UnaryOperator<String> fallback; // espeak-ng, or null: letter-to-sound instead
    private final AtomicInteger guessed = new AtomicInteger();
    private final AtomicBoolean warned = new AtomicBoolean();

    private Lexicon(Map<String, String> words, UnaryOperator<String> fallback) {
        this.words = words;
        this.fallback = fallback;
    }

    /**
     * The lexicon at {@code path}. THROWS when it cannot be read or parsed: a caller who named a
     * file and silently got a fallback has been lied to, and spelling every word out letter by
     * letter is a far worse outcome than a load failure.
     */
    public static Lexicon read(Path path, UnaryOperator<String> fallback) throws IOException {
        return new Lexicon(readTrie(Files.readAllBytes(path)), fallback);
    }

    /**
     * The lexicon bundled on the classpath, or null when the jar or image carries none. This is the
     * rung a single-file native binary lands on: it has no directory to sit beside.
     */
    public static Lexicon bundled(UnaryOperator<String> fallback) throws IOException {
        try (var in = Lexicon.class.getResourceAsStream(RESOURCE)) {
            return in == null ? null : new Lexicon(readTrie(in.readAllBytes()), fallback);
        }
    }

    /** A small in-memory lexicon, word to IPA, for tests. */
    static Lexicon of(Map<String, String> words, UnaryOperator<String> fallback) {
        return new Lexicon(Map.copyOf(words), fallback);
    }

    /**
     * IPA for one punctuation-free run of words. Every word known: their IPA, joined. An unknown
     * word and a fallback: the WHOLE run to the fallback - stress is contextual, so per-word
     * fallback would stamp a primary stress on every function word. No fallback: the unknown word
     * is guessed by {@link LetterToSound} and the guess count is logged once, because a wrong
     * pronunciation should be audible work, not an invisible gap.
     *
     * <p>Only the lexicon KEY is lowercased; the fallback sees the spelling the writer used. espeak
     * reads capitals as information: it says "GraalVM" as "graal vee em" and "graalvm" as one
     * mangled word.
     */
    public String ipa(String run) {
        String[] tokens = run.split(" ");
        String[] ipa = new String[tokens.length];
        boolean unknown = false;
        for (int i = 0; i < tokens.length; i++) {
            ipa[i] = lookup(tokens[i].toLowerCase(Locale.ROOT));
            unknown |= ipa[i] == null;
        }
        if (unknown && fallback != null) return fallback.apply(run);
        for (int i = 0; i < tokens.length; i++) {
            if (ipa[i] != null) continue;
            ipa[i] = LetterToSound.guess(tokens[i]);
            guessed.incrementAndGet();
        }
        if (guessed.get() > 0 && warned.compareAndSet(false, true))
            System.getLogger("jinfer.inflect2")
                    .log(
                            System.Logger.Level.INFO,
                            "{0} word(s) are not in the lexicon; pronounced by letter-to-sound"
                                    + " rules. Correct one with a word override, or install"
                                    + " espeak-ng to cover them all",
                            guessed.get());
        return String.join(" ", ipa);
    }

    /** Exact match, else a known suffix split off a stem that is in the lexicon. */
    private String lookup(String word) {
        String exact = words.get(word);
        if (exact != null) return exact;
        for (Suffix suffix : SUFFIXES) {
            if (!word.endsWith(suffix.ending()) || word.length() <= suffix.ending().length() + 1)
                continue;
            String stem =
                    word.substring(0, word.length() - suffix.ending().length()) + suffix.stemTail();
            String stemIpa = words.get(stem);
            String suffixIpa = words.get(suffix.spoken());
            if (stemIpa == null || suffixIpa == null) continue;
            return stemIpa + suffixIpa;
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
     * then the word's symbol ids, mapped back to IPA through {@link Inflect2#SYMBOLS}. Sorted order
     * is what makes the prefix reuse work.
     */
    private static Map<String, String> readTrie(byte[] data) throws IOException {
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

            Map<String, String> lexicon = HashMap.newHashMap(count);
            buffer.position((int) entriesAt);
            byte[] word = new byte[MAX_WORD];
            int length = 0;
            var ipa = new StringBuilder();
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
                ipa.setLength(0);
                for (int i = 0; i < symbols; i++) {
                    int id = Byte.toUnsignedInt(buffer.get());
                    if (id >= Inflect2.SYMBOLS.size())
                        throw new IOException(
                                "lexicon entry " + entry + " has invalid symbol " + id);
                    ipa.append(Inflect2.SYMBOLS.get(id));
                }
                lexicon.put(new String(word, 0, length, StandardCharsets.US_ASCII), ipa.toString());
            }
            return lexicon;
        } catch (BufferUnderflowException | IllegalArgumentException e) {
            throw new IOException("truncated or malformed lexicon", e);
        }
    }
}
