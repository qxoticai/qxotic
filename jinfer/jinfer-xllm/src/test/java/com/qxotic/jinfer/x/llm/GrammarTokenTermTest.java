package com.qxotic.jinfer.x.llm;

import static com.qxotic.jinfer.x.llm.Grammar.Term.alt;
import static com.qxotic.jinfer.x.llm.Grammar.Term.gbnf;
import static com.qxotic.jinfer.x.llm.Grammar.Term.rep;
import static com.qxotic.jinfer.x.llm.Grammar.Term.seq;
import static com.qxotic.jinfer.x.llm.Grammar.Term.text;
import static com.qxotic.jinfer.x.llm.Grammar.Term.token;
import static com.qxotic.jinfer.x.llm.TestLogits.*;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.memory.MemoryView;
import java.nio.charset.StandardCharsets;
import org.junit.jupiter.api.Test;

/**
 * Token-identity terminals ({@link Grammar.Term.Token}): the one terminal that names a vocabulary
 * id directly, letting a TRUSTED grammar span control tokens while byte grammars (and therefore
 * content) still cannot express them at all. The laws pinned here carry the reply-language design:
 * identity slots are the only admission for their id, an unexpected special stays a harmless no-op,
 * and a token's IDENTITY and its BYTE VIEW are independent match paths.
 */
public final class GrammarTokenTermTest {

    /** ids 0..7 plain bytes, 8 a MISTYPED special (non-empty bytes), 9/10 true specials. */
    static final class V implements Grammar.Vocab {
        static final String[] W = {
            "a", "b", "x", "y", "\"", "<e", "os>", ",", "<eos>", "", "", "ab"
        };
        static final int QUOTE = 9, OTHER = 10, MISTYPED = 8;

        @Override
        public int size() {
            return W.length;
        }

        @Override
        public byte[] bytes(int t) {
            return W[t].getBytes(StandardCharsets.UTF_8);
        }
    }

    static final V VOCAB = new V();

    static Grammar.Cursor cursor(Grammar.Term root) {
        return Grammar.of(root, VOCAB).cursor();
    }

    /** The admission set as a boolean per id, via the public maskLogits face. */
    static boolean[] admitted(Grammar.Cursor c) {
        MemoryView<?> logits = view(VOCAB.size());
        for (int i = 0; i < VOCAB.size(); i++) set(logits, i, 0f);
        c.maskLogits(logits);
        boolean[] ok = new boolean[VOCAB.size()];
        for (int i = 0; i < VOCAB.size(); i++) ok[i] = get(logits, i) == 0f;
        return ok;
    }

    static void assertOnly(Grammar.Cursor c, int... ids) {
        boolean[] ok = admitted(c);
        for (int i = 0; i < ok.length; i++) {
            boolean expected = false;
            for (int id : ids) expected |= id == i;
            assertTrue(
                    ok[i] == expected,
                    "token " + i + " (" + V.W[i] + ") admitted=" + ok[i] + ", want " + expected);
        }
    }

    @Test
    void identityIsTheOnlyAdmissionAtItsPosition() {
        Grammar.Cursor c = cursor(seq(text("a"), token(V.QUOTE), text("b")));
        assertOnly(c, 0); // just 'a': specials are NOT admissible off-position (non-accepting)
        c.advanceWith(0);
        assertOnly(c, V.QUOTE); // a FORCED state: exactly one admissible token, by identity
        c.advanceWith(V.QUOTE);
        assertOnly(c, 1);
        c.advanceWith(1);
        assertTrue(c.exhausted());
        // at the exhausted accept state the legacy law returns: every empty-byte token may end
        boolean[] ok = admitted(c);
        assertTrue(ok[V.QUOTE] && ok[V.OTHER]);
        assertFalse(ok[0] || ok[V.MISTYPED]);
    }

    @Test
    void anUnexpectedSpecialNeverAdvancesAndNeverKills() {
        Grammar.Cursor c = cursor(seq(text("a"), token(V.QUOTE), text("b")));
        c.advanceWith(0);
        c.advanceWith(V.OTHER); // the WRONG special: the dead-end advanceWith(eos) contract
        assertOnly(c, V.QUOTE); // state unchanged - not advanced, not dead
        c.advanceWith(V.QUOTE);
        assertOnly(c, 1);
    }

    @Test
    void anImpossibleByteTokenStillKills() {
        Grammar.Cursor c = cursor(token(V.QUOTE));
        c.advanceWith(0); // a byte token where only an identity fits: dead state, loudly
        MemoryView<?> logits = view(VOCAB.size());
        for (int i = 0; i < VOCAB.size(); i++) set(logits, i, 0f);
        assertFalse(c.maskLogits(logits), "a dead cursor must admit nothing");
    }

    @Test
    void exhaustedAfterATrailingIdentityTerminal() {
        Grammar.Cursor c = cursor(seq(text("a"), token(V.QUOTE)));
        c.advanceWith(0);
        assertFalse(c.exhausted());
        c.advanceWith(V.QUOTE);
        assertTrue(c.exhausted()); // the release point: prefix pins end on identity closes
    }

    @Test
    void aPinnedMistypedIdAdmitsByIdentityWhateverItsBytes() {
        // Gemma4's <eos> is typed NORMAL in the GGUF, so its byte view is the 5-char string;
        // an identity terminal admits the ID regardless - no byte path exists here at all
        Grammar.Cursor c = cursor(token(V.MISTYPED));
        assertOnly(c, V.MISTYPED);
        c.advanceWith(V.MISTYPED);
        assertTrue(c.exhausted());
    }

    @Test
    void identityAndBytePathsUnionWhenBothFit() {
        // token 8 IS "<eos>" bytes AND id 8: alternative one names the id, alternative two
        // spells the bytes - advancing with 8 keeps both interpretations alive
        Grammar.Cursor c =
                cursor(
                        alt(
                                seq(token(V.MISTYPED), text("x")),
                                seq(text("<e"), text("os>"), text("y"))));
        assertOnly(c, V.MISTYPED, 5); // the id, and the '<e' byte prefix
        c.advanceWith(V.MISTYPED);
        assertOnly(c, 2, 3); // both continuations live: x (identity path) and y (byte path)
        c.advanceWith(3);
        assertTrue(c.exhausted());
    }

    @Test
    void gbnfFragmentsEmbedWholeWithShiftedRuleIds() {
        Grammar.Cursor c =
                cursor(
                        seq(
                                token(V.QUOTE),
                                gbnf("root ::= item (\",\" item)*\nitem ::= \"a\" | \"b\""),
                                token(V.QUOTE)));
        c.advanceWith(V.QUOTE);
        assertOnly(c, 0, 1); // the fragment's own first set - never ','
        c.advanceWith(0);
        c.advanceWith(7); // ","
        assertOnly(c, 0, 1);
        c.advanceWith(1);
        c.advanceWith(V.QUOTE);
        assertTrue(c.exhausted());
    }

    @Test
    void twoFragmentsKeepTheirRuleIdsApart() {
        Grammar.Cursor c = cursor(seq(gbnf("root ::= \"a\""), gbnf("root ::= \"b\"")));
        c.advanceWith(0);
        c.advanceWith(1);
        assertTrue(c.exhausted());
    }

    @Test
    void aContentQuoteCanNeverCloseAnIdentityQuotedString() {
        // the CPT string law: the delimiter is the quote TOKEN; a '"' typed as CONTENT is just
        // bytes and must stay inside the string - injection-proof by construction
        Grammar.Cursor c =
                cursor(
                        seq(
                                token(V.QUOTE),
                                rep(alt(text("a"), text("b"), text("\"")), 0, -1),
                                token(V.QUOTE)));
        c.advanceWith(V.QUOTE);
        c.advanceWith(4); // a content '"' (byte token)
        assertFalse(c.exhausted(), "a content quote must not close the string");
        c.advanceWith(0);
        c.advanceWith(V.QUOTE); // the quote TOKEN closes
        assertTrue(c.exhausted());
    }

    @Test
    void tryAdvanceReportsTheTriState() {
        Grammar.Cursor c = cursor(seq(text("a"), token(V.QUOTE)));
        assertTrue(c.tryAdvance(0), "a fitting byte token consumes");
        assertFalse(c.tryAdvance(V.OTHER), "the wrong special is a no-op: not consumed");
        assertTrue(c.tryAdvance(V.QUOTE), "the state survived the no-op");
        assertTrue(c.exhausted());

        Grammar.Cursor dead = cursor(token(V.QUOTE));
        assertFalse(dead.tryAdvance(0), "an impossible byte token kills");
        assertFalse(dead.tryAdvance(V.QUOTE), "and the cursor stays dead");
    }

    @Test
    void aByteTokenNeverCrossesAnIdentityBoundary() {
        // the byte-boundary dual of the injection law: "ab" spans text("a") token(Q) text("b"),
        // but an identity slot matches no byte - the merged token must be inadmissible
        Grammar.Cursor c = cursor(seq(text("a"), token(V.QUOTE), text("b")));
        boolean[] ok = admitted(c);
        assertFalse(ok[11], "the merged 'ab' token would tunnel through the identity slot");
        assertTrue(ok[0]);
    }

    @Test
    void anOutOfVocabularyIdFailsAtCompileNotAsADeadGrammar() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Grammar.of(seq(text("a"), token(999)), VOCAB),
                "a stale pinned id must fail loudly, never compile to a mask that admits nothing");
        assertThrows(
                IllegalArgumentException.class,
                () -> Grammar.of(rep(token(999), 0, -1), VOCAB),
                "inside a repetition too");
    }

    @Test
    void repetitionOverIdentityTerminals() {
        Grammar.Cursor c = cursor(rep(token(V.QUOTE), 1, 2));
        assertOnly(c, V.QUOTE); // min 1: not yet accepting, so no other special sneaks in
        c.advanceWith(V.QUOTE);
        boolean[] ok = admitted(c);
        assertTrue(ok[V.QUOTE], "a second repetition is admissible");
        assertTrue(ok[V.OTHER], "the state accepts, so any special may end it");
        c.advanceWith(V.QUOTE);
        assertTrue(c.exhausted());
    }
}
