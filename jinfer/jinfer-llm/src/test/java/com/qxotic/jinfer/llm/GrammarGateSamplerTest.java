package com.qxotic.jinfer.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import org.junit.jupiter.api.Test;

/**
 * The grammar gate's four phases, driven deterministically with synthetic logits (no model): the
 * undecided first token (grammar-decided answer OR the right to open a think span), the dormant
 * think span, the newline skip, and full constraint - plus the no-markers immediate path and the
 * complete-grammar forced stop. This is the direct coverage for the gate redesign that fixed the
 * dormant-forever bug (a think-capable vocab on a model that answers directly).
 */
class GrammarGateSamplerTest {

    static final GrammarSpecTest.ByteVocab BV = GrammarSpecTest.BV;
    static final int EOS = GrammarSpecTest.eosId(BV);
    static final int OPEN = '<'; // stand-in think-open marker (any id outside the grammar)
    static final int CLOSE = '>'; // stand-in think-close
    static final int NL = '\n';

    /** Deterministic inner sampler: argmax. */
    static final Sampler ARGMAX =
            logits -> {
                int best = 0;
                float bestV = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < logits.size(); i++) {
                    float v = logits.getFloat(i);
                    if (v > bestV) {
                        bestV = v;
                        best = i;
                    }
                }
                return best;
            };

    /** Logits of vocab size, all strongly negative except the favorites (descending). */
    static FloatTensor favoring(int... ids) {
        F32FloatTensor logits = F32FloatTensor.allocate(BV.size());
        for (int i = 0; i < BV.size(); i++) logits.setFloat(i, -100f);
        float v = 10f;
        for (int id : ids) logits.setFloat(id, v -= 1f);
        return logits;
    }

    static Sampler gate(String gbnf, boolean startInThink) {
        return Sampler.withGrammar(
                ARGMAX, Grammar.of(gbnf, BV).cursor(), EOS, OPEN, CLOSE, startInThink, new int[] {NL});
    }

    @Test
    void undecidedDirectAnswerIsConstrainedFromTokenZero() {
        Sampler s = gate("root ::= \"yes\" | \"no\"", false);
        // the model 'wants' X (invalid); the grammar must decide the very first token
        assertEquals('y', s.sampleToken(favoring('X', 'y')));
        assertEquals('e', s.sampleToken(favoring('X', 'e')));
        assertEquals('s', s.sampleToken(favoring('X', 's')));
        assertEquals(EOS, s.sampleToken(favoring('X')), "complete sentence forces the stop");
    }

    @Test
    void undecidedKeepsTheRightToReason() {
        Sampler s = gate("root ::= \"yes\" | \"no\"", false);
        // the model prefers to open a think span: the union must let it through
        assertEquals(OPEN, s.sampleToken(favoring(OPEN, 'X', 'y')));
        // inside the span: fully unconstrained (Q is not in the grammar's language)
        assertEquals('Q', s.sampleToken(favoring('Q')));
        assertEquals(CLOSE, s.sampleToken(favoring(CLOSE)));
        // skip phase: the boilerplate newline passes without touching the grammar
        assertEquals(NL, s.sampleToken(favoring(NL, 'X')));
        // first answer token: constrained again
        assertEquals('n', s.sampleToken(favoring('X', 'n')));
        assertEquals('o', s.sampleToken(favoring('X', 'o')));
        assertEquals(EOS, s.sampleToken(favoring('X')));
    }

    @Test
    void startInThinkStaysDormantUntilTheClose() {
        Sampler s = gate("root ::= \"yes\" | \"no\"", true);
        // prompt-opened span: the first tokens are reasoning, never constrained
        assertEquals('Q', s.sampleToken(favoring('Q', 'y')));
        assertEquals('R', s.sampleToken(favoring('R')));
        assertEquals(CLOSE, s.sampleToken(favoring(CLOSE)));
        assertEquals('y', s.sampleToken(favoring('X', 'y')), "constrained after the close");
    }

    @Test
    void noMarkersConstrainImmediately() {
        Sampler s =
                Sampler.withGrammar(
                        ARGMAX,
                        Grammar.of("root ::= \"no\"", BV).cursor(),
                        EOS,
                        -1,
                        -1,
                        false,
                        null);
        assertEquals('n', s.sampleToken(favoring('X', 'n')));
        assertEquals('o', s.sampleToken(favoring('X', 'o')));
    }

    @Test
    void skipPhaseOnlySkipsTheDeclaredTokens() {
        Sampler s = gate("root ::= \"yes\"", true);
        assertEquals(CLOSE, s.sampleToken(favoring(CLOSE)));
        // NOT a skip token after the close: the grammar decides it immediately
        assertEquals('y', s.sampleToken(favoring('X', 'y')));
    }

    @Test
    void unionDoesNotLeakBeyondTheFirstToken() {
        Sampler s = gate("root ::= \"yes\"", false);
        assertEquals('y', s.sampleToken(favoring('X', 'y'))); // decided: grammar path
        // OPEN was only unioned at token zero; now it must be masked like anything else
        assertEquals('e', s.sampleToken(favoring(OPEN, 'X', 'e')));
        assertTrue(true);
    }
}
