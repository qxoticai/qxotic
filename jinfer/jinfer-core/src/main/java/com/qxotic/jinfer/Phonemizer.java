package com.qxotic.jinfer;

import java.util.List;
import java.util.function.UnaryOperator;

/**
 * Text to phoneme ids in ONE model's phoneme vocabulary - the speech counterpart of a tokenizer.
 * The ids mean something only to the model whose symbol table built the phonemizer, exactly as
 * token ids mean something only to the model whose vocabulary built the tokenizer.
 *
 * <p>Like a tokenizer, a phonemizer does not carry the model's text policy: normalization, sentence
 * chunking and a terminating mark are {@link SpeechSynthesisModel#speak}'s, applied before the text
 * reaches it. Feed it what you would feed a tokenizer.
 */
@FunctionalInterface
public interface Phonemizer {

    /** Phoneme ids for {@code text}. Never null; empty when nothing was pronounceable. */
    int[] phonemize(String text);

    /**
     * A phonemizer for the IPA code-point family (StyleTTS2, VITS, Kokoro, Inflect): one Unicode
     * code point per symbol, and a grapheme-to-phoneme step that emits IPA text. A model whose
     * symbols are not single code points - ARPAbet's {@code AH0} - implements the interface
     * directly.
     *
     * <p>{@code symbols} is the model's own table, indexed by id: one code point per entry, and ""
     * for a slot the model never emits. {@code ipa} sees one punctuation-free RUN of words at a
     * time and returns its IPA; the marks between runs are re-emitted as symbols of their own, in
     * order, and IPA the table lacks is dropped. A run rather than a word because stress is
     * contextual: given words one at a time, espeak stamps a primary stress on every function word
     * and can never flap across a boundary.
     */
    static Phonemizer ipa(List<String> symbols, UnaryOperator<String> ipa) {
        return new RunPhonemizer(symbols, ipa);
    }
}
