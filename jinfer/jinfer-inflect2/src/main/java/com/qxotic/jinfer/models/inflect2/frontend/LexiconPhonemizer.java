// Pure-Java phonemizer — IVL2 lexicon.bin trie with morphology fallback.
// Normalizes text, then does word-by-word lookup in the trie.
package com.qxotic.jinfer.models.inflect2.frontend;

import com.qxotic.jinfer.models.inflect2.Symbols;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

final class LexiconPhonemizer implements Phonemizer {
    private static final int SPACE = 16;

    private final Map<String, int[]> dict;

    private LexiconPhonemizer(Map<String, int[]> dict) {
        this.dict = dict;
    }

    static LexiconPhonemizer tryCreate() {
        for (String p :
                new String[] {
                    "lexicon.bin",
                    "../Inflect-Nano-v2/lexicon.bin",
                    System.getProperty("user.home") + "/.inflect/lexicon.bin"
                }) {
            try {
                var path = Path.of(p);
                if (Files.exists(path) && Files.size(path) > 1000) {
                    System.err.println("[LexiconPhonemizer] " + path);
                    return new LexiconPhonemizer(readTrie(Files.readAllBytes(path)));
                }
            } catch (IOException ignored) {
            }
        }
        try (var in = LexiconPhonemizer.class.getResourceAsStream("/lexicon.bin")) {
            if (in != null) {
                var dict = readTrie(in.readAllBytes());
                if (!dict.isEmpty()) return new LexiconPhonemizer(dict);
            }
        } catch (Exception ignored) {
        }
        return null;
    }

    @Override
    public int[] phonemize(String text) {
        String norm = TextNormalizer.normalize(text);
        var syms = new ArrayList<Integer>();
        for (String word : norm.split("\\s+")) {
            if (word.isEmpty()) continue;
            int[] ids = lookupWord(word.toLowerCase());
            if (ids != null && ids.length > 0) {
                for (int id : ids) syms.add(id);
                syms.add(SPACE);
            }
        }
        if (!syms.isEmpty() && syms.get(syms.size() - 1) == SPACE) syms.remove(syms.size() - 1);
        return Symbols.blankIntersperse(syms.stream().mapToInt(Integer::intValue).toArray());
    }

    private int[] lookupWord(String lower) {
        int[] ids = dict.get(lower);
        if (ids != null) return ids;
        for (int m = 0; m < MORPH; m++) {
            String end = MORPH_END[m], stemFix = MORPH_STEM[m], sfxWord = MORPH_SFX[m];
            if (lower.endsWith(end) && lower.length() > end.length() + 1) {
                String stem = lower.substring(0, lower.length() - end.length());
                if (!stemFix.isEmpty()) stem += stemFix;
                int[] stemIds = dict.get(stem);
                int[] sfxIds = sfxWord.isEmpty() ? null : dict.get(sfxWord);
                if (stemIds != null && sfxIds != null) {
                    int[] r = new int[stemIds.length + sfxIds.length];
                    System.arraycopy(stemIds, 0, r, 0, stemIds.length);
                    System.arraycopy(sfxIds, 0, r, stemIds.length, sfxIds.length);
                    return r;
                }
            }
        }
        return null;
    }

    private static final String[] MORPH_END = {"ies", "'s", "s'", "es", "s", "ied", "ed", "ing"};
    private static final String[] MORPH_STEM = {"y", "", "", "", "", "y", "", ""};
    private static final String[] MORPH_SFX = {"z", "z", "z", "z", "z", "d", "d", "ng"};
    private static final int MORPH = MORPH_END.length;

    // ── trie reader ──────────────────────────────────────────────────────

    private static Map<String, int[]> readTrie(byte[] data) {
        var buf = java.nio.ByteBuffer.wrap(data).order(java.nio.ByteOrder.LITTLE_ENDIAN);
        byte[] magic = new byte[4];
        buf.get(magic);
        if (!"IVL2".equals(new String(magic)) || buf.getShort() != 1) return Map.of();
        buf.position(buf.position() + 2 + 32);
        long count = Integer.toUnsignedLong(buf.getInt());
        buf.position(buf.position() + 8);
        long off = buf.getLong();
        buf.position(buf.position() + 8);

        var dict = new HashMap<String, int[]>((int) (count * 4 / 3));
        buf.position((int) off);
        byte[] cur = new byte[64];
        int len = 0;
        for (long i = 0; i < count && buf.hasRemaining(); i++) {
            int pf = Byte.toUnsignedInt(buf.get());
            int sf = Byte.toUnsignedInt(buf.get());
            if (pf > len) break;
            len = pf + sf;
            if (len > cur.length) cur = Arrays.copyOf(cur, len * 2);
            if (sf > 0) buf.get(cur, pf, sf);
            int tc = Byte.toUnsignedInt(buf.get());
            if (tc > 0) {
                int[] tok = new int[tc];
                for (int t = 0; t < tc; t++) tok[t] = Byte.toUnsignedInt(buf.get());
                dict.put(new String(cur, 0, len), tok);
            }
        }
        return dict;
    }
}
