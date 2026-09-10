package com.qxotic.jinfer.llm;

import static com.qxotic.jinfer.llm.TestLogits.*;

import com.qxotic.jota.memory.MemoryView;
import java.nio.charset.StandardCharsets;

/** Membership oracle: EOS must be allowed after every input byte has been consumed. */
final class GrammarMembership {
    private GrammarMembership() {}

    /** 256 byte tokens plus an empty-byte EOS. */
    static final Grammar.Vocab BV =
            new Grammar.Vocab() {
                public int size() {
                    return 257;
                }

                public byte[] bytes(int token) {
                    return token == 256 ? new byte[0] : new byte[] {(byte) token};
                }
            };

    static int eosId(Grammar.Vocab vocab) {
        for (int i = 0; i < vocab.size(); i++) if (vocab.bytes(i).length == 0) return i;
        return -1;
    }

    /** First rejected byte (-1 if none), followed by whether the complete input is accepted. */
    static int[] probe(Grammar.Spec spec, Grammar.Vocab vocab, byte[] bytes) {
        Grammar.Cursor cursor = spec.cursor();
        MemoryView<?> logits = view(vocab.size());
        int eos = eosId(vocab);
        for (int i = 0; i < bytes.length; i++) {
            mask(cursor, logits, vocab.size());
            int b = bytes[i] & 255;
            if (!allowed(logits, b)) return new int[] {i, 0};
            cursor.advanceWith(b);
        }
        mask(cursor, logits, vocab.size());
        return new int[] {-1, allowed(logits, eos) ? 1 : 0};
    }

    static void mask(Grammar.Cursor cursor, MemoryView<?> logits, int size) {
        for (int i = 0; i < size; i++) set(logits, i, 0f);
        cursor.maskLogits(logits);
    }

    static boolean allowed(MemoryView<?> logits, int id) {
        return id >= 0 && get(logits, id) > -1e30f;
    }

    static boolean accepts(Grammar.Spec spec, Grammar.Vocab vocab, String text) {
        int[] result = probe(spec, vocab, text.getBytes(StandardCharsets.UTF_8));
        return result[0] == -1 && result[1] == 1;
    }

    static boolean notMember(Grammar.Spec spec, Grammar.Vocab vocab, String text) {
        return !accepts(spec, vocab, text);
    }

    static int rejectAt(Grammar.Spec spec, Grammar.Vocab vocab, String text) {
        return probe(spec, vocab, text.getBytes(StandardCharsets.UTF_8))[0];
    }

    static boolean validPrefix(Grammar.Spec spec, Grammar.Vocab vocab, String text) {
        return rejectAt(spec, vocab, text) == -1;
    }
}
