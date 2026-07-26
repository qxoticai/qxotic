package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.llm.Grammar;
import com.qxotic.jinfer.llm.Sampler;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * The channel-scoped constraint, driven with a SCRIPTED parser (channels are stated, not inferred
 * from token ids - even more deterministic than the old gate tests): free reasoning and structure,
 * output constrained from token zero, specials allowed in output without advancing the cursor,
 * newline tolerance before the grammar starts, per-channel independent cursors, and the
 * derived-advance rule that survives pending UTF-8 (it never looks at fragments).
 */
class ChannelConstrainedSamplerTest {

    /** Byte-token vocab: id == byte value; one empty token as EOS. */
    static final Grammar.Vocab BV =
            new Grammar.Vocab() {
                @Override
                public int size() {
                    return 257;
                }

                @Override
                public byte[] bytes(int id) {
                    return id == 256 ? new byte[0] : new byte[] {(byte) id};
                }
            };

    static final int EOS = 256;
    static final int SPECIAL = '<'; // stand-in special (span close, turn end...)
    static final int NL = '\n';

    static final Sampler ARGMAX =
            logits -> {
                int best = 0;
                float bestV = Float.NEGATIVE_INFINITY;
                for (int i = 0; i < logits.size(); i++) {
                    if (logits.getFloat(i) > bestV) {
                        bestV = logits.getFloat(i);
                        best = i;
                    }
                }
                return best;
            };

    static FloatTensor favoring(int... ids) {
        F32FloatTensor logits = F32FloatTensor.allocate(BV.size());
        for (int i = 0; i < BV.size(); i++) logits.setFloat(i, -100f);
        float v = 10f;
        for (int id : ids) logits.setFloat(id, v -= 1f);
        return logits;
    }

    /** A parser whose pendingChannel sequence is SCRIPTED; feeds are recorded, not interpreted. */
    static final class ScriptedParser implements ReplyParser {
        final List<String> script; // "null" spells the structure region
        int at;

        ScriptedParser(String... channels) {
            this.script = List.of(channels);
        }

        @Override
        public String pendingChannel() {
            String c = script.get(Math.min(at, script.size() - 1));
            return c.equals("null") ? null : c;
        }

        @Override
        public String feed(int token) {
            at++;
            return "";
        }

        @Override
        public boolean reasoning() {
            return false;
        }

        @Override
        public Set<String> outputChannels() {
            return Set.of("content");
        }

        @Override
        public Message finish() {
            return new Message(Role.ASSISTANT, "");
        }
    }

    static Sampler sampler(ReplyParser parser, Map<String, Grammar.Cursor> byChannel) {
        return new ChannelConstrainedSampler(
                ARGMAX,
                parser,
                byChannel::get,
                t -> t == SPECIAL,
                new int[] {SPECIAL},
                new int[] {NL},
                EOS);
    }

    static Grammar.Cursor yesNo() {
        return Grammar.of("root ::= \"yes\" | \"no\"", BV).cursor();
    }

    @Test
    void outputConstrainedFromTokenZero() {
        var s =
                sampler(
                        new ScriptedParser("content", "content", "content", "content"),
                        Map.of("content", yesNo()));
        assertEquals('y', s.sampleToken(favoring('X', 'y')));
        assertEquals('e', s.sampleToken(favoring('X', 'e')));
        assertEquals('s', s.sampleToken(favoring('X', 's')));
        int end = s.sampleToken(favoring('X'));
        assertTrue(
                end == EOS || end == SPECIAL,
                "complete grammar admits only an ending (grammar EOS or a turn-end special): "
                        + end);
    }

    @Test
    void reasoningAndStructureAreFree() {
        var s =
                sampler(
                        new ScriptedParser("reasoning", "null", "tool-call", "content"),
                        Map.of("content", yesNo()));
        assertEquals('Q', s.sampleToken(favoring('Q', 'y')), "reasoning free");
        assertEquals('R', s.sampleToken(favoring('R')), "structure free");
        assertEquals('Z', s.sampleToken(favoring('Z')), "call payload free");
        assertEquals('n', s.sampleToken(favoring('X', 'n')), "output constrained");
    }

    @Test
    void specialsStayLegalInOutputAndNeverAdvance() {
        var s =
                sampler(
                        new ScriptedParser("content", "content", "content"),
                        Map.of("content", yesNo()));
        // the model closes a span / ends the turn mid-output: the union must allow it
        assertEquals(SPECIAL, s.sampleToken(favoring(SPECIAL, 'X')));
        // and the cursor must NOT have advanced: 'y' still starts the grammar
        assertEquals('y', s.sampleToken(favoring('X', 'y')));
        assertEquals('e', s.sampleToken(favoring('X', 'e')));
    }

    @Test
    void newlineToleranceOnlyDirectlyAfterStructure() {
        var s =
                sampler(
                        new ScriptedParser(
                                "reasoning", "content", "content", "content", "content", "content"),
                        Map.of("content", yesNo()));
        assertEquals(SPECIAL, s.sampleToken(favoring(SPECIAL)), "span close (free channel)");
        assertEquals(NL, s.sampleToken(favoring(NL, 'X')), "boilerplate newline passes");
        assertEquals(NL, s.sampleToken(favoring(NL, 'X')), "a second newline passes");
        // the cap: a THIRD newline is no longer boilerplate - the grammar decides
        assertEquals('y', s.sampleToken(favoring(NL, 'X', 'y')), "bounded by construction");
        // grammar started: a newline is now just an invalid token, masked away
        assertEquals('e', s.sampleToken(favoring(NL, 'X', 'e')));
    }

    @Test
    void noNewlineToleranceAtAColdStart() {
        // no preceding structure: a model that LIKES newlines must not free-run - the grammar
        // decides token zero (the 200-blank-lines regression this rule fixed)
        var s = sampler(new ScriptedParser("content", "content"), Map.of("content", yesNo()));
        assertEquals('y', s.sampleToken(favoring(NL, 'X', 'y')));
    }

    @Test
    void escapeIsOneShot() {
        // the LFM2.5 regression: with an exhausted reasoning cap, a live escape let the model
        // cycle open/force-close/newlines to the token budget - once reasoning has happened,
        // the open marker may not be offered again
        var s =
                sampler(
                        new ScriptedParser("content", "reasoning", "reasoning", "content"),
                        Map.of("content", yesNo()));
        assertEquals(SPECIAL, s.sampleToken(favoring(SPECIAL, 'y')), "escape: the model reasons");
        assertEquals('R', s.sampleToken(favoring('R')), "reasoning free (spends the escape)");
        assertEquals(SPECIAL, s.sampleToken(favoring(SPECIAL)), "span close (free channel)");
        // the model still PREFERS reopening - the retired escape forces the grammar instead
        assertEquals('y', s.sampleToken(favoring(SPECIAL, NL, 'y')), "no reopen: grammar decides");
    }

    @Test
    void perChannelCursorsAreIndependent() {
        var s =
                sampler(
                        new ScriptedParser("content", "aux", "content", "aux"),
                        Map.of(
                                "content",
                                yesNo(),
                                "aux",
                                Grammar.of("root ::= \"ok\"", BV).cursor()));
        assertEquals('y', s.sampleToken(favoring('X', 'y', 'o'))); // content cursor: y
        assertEquals('o', s.sampleToken(favoring('X', 'y', 'o'))); // aux cursor: o
        assertEquals('e', s.sampleToken(favoring('X', 'e', 'k'))); // content continues: yes
        assertEquals('k', s.sampleToken(favoring('X', 'e', 'k'))); // aux continues: ok
    }

    @Test
    void advanceDerivationIgnoresFragments() {
        // the scripted parser ALWAYS returns empty fragments - if advancement depended on
        // fragment text (the pending-UTF-8 lie), the grammar would never progress
        var s =
                sampler(
                        new ScriptedParser("content", "content", "content"),
                        Map.of("content", yesNo()));
        assertEquals('n', s.sampleToken(favoring('X', 'n')));
        assertEquals('o', s.sampleToken(favoring('X', 'o')));
        int end = s.sampleToken(favoring('X'));
        assertTrue(end == EOS || end == SPECIAL, "an ending after completion: " + end);
    }
}
