package com.qxotic.toknroll;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.text.Normalizer.Form;
import org.junit.jupiter.api.Test;

class NormalizerComponentTest {

    @Test
    void identityReturnsInputReference() {
        CharSequence input = new StringBuilder("A\u0301BC");

        CharSequence out = Normalizer.identity().apply(input);

        assertSame(input, out);
    }

    @Test
    void lowercaseUsesRootLocale() {
        String input = "I ISTANBUL";

        CharSequence out = Normalizer.lowercase().apply(input);

        assertEquals("i istanbul", out.toString());
    }

    @Test
    void unicodeNfcComposesDecomposedCharacters() {
        String decomposed = "e\u0301";

        CharSequence out = Normalizer.unicode(Form.NFC).apply(decomposed);

        assertEquals("é", out.toString());
    }

    @Test
    void sequenceAppliesInOrder() {
        Normalizer sequence =
                Normalizer.sequence(
                        text -> "[" + text + "]", Normalizer.lowercase(), text -> text + "!");

        CharSequence out = sequence.apply("HeLLo");

        assertEquals("[hello]!", out.toString());
    }

    @Test
    void emptySequenceIsIdentity() {
        CharSequence input = new StringBuilder("abc");

        CharSequence out = Normalizer.sequence().apply(input);

        assertSame(input, out);
    }

    @Test
    void sequenceRejectsNullArrayAndNullMember() {
        assertThrows(NullPointerException.class, () -> Normalizer.sequence((Normalizer[]) null));

        Normalizer sequence = Normalizer.sequence(Normalizer.identity(), null);
        assertThrows(NullPointerException.class, () -> sequence.apply("x"));
    }

    @Test
    void unicodeRejectsNullForm() {
        assertThrows(NullPointerException.class, () -> Normalizer.unicode(null));
    }
}
