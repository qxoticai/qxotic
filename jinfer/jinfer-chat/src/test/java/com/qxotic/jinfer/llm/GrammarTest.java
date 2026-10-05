package com.qxotic.jinfer.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

final class GrammarTest {
    private static final Grammar.Vocab VOCAB =
            new Grammar.Vocab() {
                private final String[] tokens = {"(", ")", "x", ""};

                public int size() {
                    return tokens.length;
                }

                public byte[] bytes(int token) {
                    return tokens[token].getBytes(StandardCharsets.UTF_8);
                }
            };

    @Test
    void recursiveGrammarMasksMemoryViewLogits() {
        Grammar.Cursor cursor = Grammar.of("root ::= \"(\" root \")\" | \"x\"", VOCAB).cursor();
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> logits =
                    Views.allocateF32(MemoryAllocators.ofArena(arena), VOCAB.size());

            assertMask(cursor, logits, true, false, true, false);
            cursor.advanceWith(0);
            assertMask(cursor, logits, true, false, true, false);
            cursor.advanceWith(2);
            assertMask(cursor, logits, false, true, false, false);
            cursor.advanceWith(1);
            assertTrue(cursor.accepting());
            assertMask(cursor, logits, false, false, false, true);
        }
    }

    @Test
    void undersizedLogitsFailBeforeWriting() {
        Grammar.Cursor cursor = Grammar.of("root ::= \"x\"", VOCAB).cursor();
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> logits =
                    Views.allocateF32(MemoryAllocators.ofArena(arena), 2);
            IllegalArgumentException error =
                    Assertions.assertThrows(
                            IllegalArgumentException.class, () -> cursor.maskLogits(logits));
            assertTrue(error.getMessage().contains("vocabulary 4"));
        }
    }

    private static void assertMask(
            Grammar.Cursor cursor,
            MemoryView<MemorySegment> logits,
            boolean open,
            boolean close,
            boolean text,
            boolean eos) {
        Views.copyFromArray(logits, 0, new float[4], 0, 4, "logits");
        assertTrue(cursor.maskLogits(logits));
        float[] values = Views.toFloatArray(logits, "logits");
        assertEquals(open, Float.isFinite(values[0]));
        assertEquals(close, Float.isFinite(values[1]));
        assertEquals(text, Float.isFinite(values[2]));
        assertEquals(eos, Float.isFinite(values[3]));
        assertFalse(Float.isNaN(values[0]));
    }

    @Test
    void aSourceWithoutARuleIsAnErrorNotAPassThrough() {
        // ":=" for "::=": every other syntax error throws; this one used to compile to the
        // unmasked DISABLED spec and generate free text under a caller that believed otherwise
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Grammar.of("root := \"yes\" | \"no\"", VOCAB));
        assertTrue(e.getMessage().contains("::="), e.getMessage());
    }

    @Test
    void logitRowsPastTheVocabularyAreMasked() {
        // a padded output head: 4 tokens, 6 logit rows; rows 4 and 5 name no token
        Grammar.Cursor cursor = Grammar.of("root ::= \"x\"", VOCAB).cursor();
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> logits =
                    Views.allocateF32(MemoryAllocators.ofArena(arena), 6);
            assertTrue(cursor.maskLogits(logits));
            float[] values = Views.toFloatArray(logits, "logits");
            assertTrue(Float.isFinite(values[2]));
            assertEquals(Float.NEGATIVE_INFINITY, values[4]);
            assertEquals(Float.NEGATIVE_INFINITY, values[5]);
        }
    }

    @Test
    void malformedGbnfIsRefusedWithItsPosition() {
        for (String source :
                new String[] {
                    "root ::= \"x\" [", // unterminated class
                    "root ::= \"x\" ]", // stray character
                    "root ::= \"x\" ;",
                    "root ::= \"x\"{2", // unterminated bound
                    "root ::= \"x\"{a}", // not a bound
                    "root ::= \"x\"{3,1}", // max below min
                    "root ::= \"x\"{-1}",
                }) {
            IllegalArgumentException e =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> Grammar.of(source, GrammarMembership.BV),
                            source);
            assertTrue(e.getMessage().contains("column"), e.getMessage());
        }
        // the well-formed spellings still compile
        Grammar.Spec bounded = Grammar.of("root ::= \"x\"{1,2} \"y\" {0,0}", GrammarMembership.BV);
        assertTrue(GrammarMembership.accepts(bounded, GrammarMembership.BV, "xx"));
        assertFalse(GrammarMembership.accepts(bounded, GrammarMembership.BV, "xxx"));
    }

    @Test
    void schemaMaxItemsZeroIsTheEmptyArray() {
        Grammar.Spec spec =
                Grammar.fromSchema(
                        Map.of("type", "array", "items", Map.of("type", "integer"), "maxItems", 0),
                        GrammarMembership.BV);
        assertTrue(GrammarMembership.accepts(spec, GrammarMembership.BV, "[]"));
        assertFalse(GrammarMembership.accepts(spec, GrammarMembership.BV, "[1]"));
        assertFalse(GrammarMembership.accepts(spec, GrammarMembership.BV, "[1,2,3]"));
    }

    @Test
    void schemaMinimumAboveMaximumIsRefused() {
        IllegalArgumentException strings =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                Grammar.schemaGbnf(
                                        Map.of("type", "string", "minLength", 3, "maxLength", 1)));
        assertTrue(strings.getMessage().contains("minLength 3"), strings.getMessage());
        IllegalArgumentException arrays =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                Grammar.schemaGbnf(
                                        Map.of("type", "array", "minItems", 2, "maxItems", 0)));
        assertTrue(arrays.getMessage().contains("minItems 2"), arrays.getMessage());
    }
}
