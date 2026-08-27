package com.qxotic.jinfer.llm;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.Tokenizer;
import java.lang.ref.WeakReference;
import org.junit.jupiter.api.Test;

/** A tokenizer that compiled a grammar is collectable once its model is gone. */
class GrammarLifetimeTest {

    @Test
    void aTokenizerIsNotPinnedByItsGrammarCaches() throws Exception {
        // the caches are weak-keyed by tokenizer; their VALUE used to capture the key
        Tokenizer tokenizer = new SpecialTokensTest.FakeTokenizer();
        Grammar.Spec spec = Grammar.of("root ::= \"a\"", tokenizer);
        assertTrue(spec != null);
        WeakReference<Tokenizer> ref = new WeakReference<>(tokenizer);
        tokenizer = null;
        spec = null;
        for (int i = 0; i < 50 && ref.get() != null; i++) {
            System.gc();
            Thread.sleep(20);
        }
        assertNull(ref.get(), "the tokenizer stayed reachable through the grammar caches");
    }
}
