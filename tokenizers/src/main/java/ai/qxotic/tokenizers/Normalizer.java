package ai.qxotic.tokenizers;

import java.text.Normalizer.Form;
import java.util.function.Function;

/**
 * A functional interface for text normalization operations that can be used in tokenization
 * pipelines. Normalizers transform text into a canonical form while preserving its essential
 * characteristics. This interface extends {@link Function} to operate on {@link CharSequence}
 * inputs and outputs.
 *
 * <p>Common use cases include:
 *
 * <ul>
 *   <li>Unicode normalization (NFC, NFD, NFKC, NFKD)
 *   <li>Case normalization
 *   <li>Whitespace normalization
 *   <li>Character substitution
 * </ul>
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * Normalizer nfkcNormalizer = Normalizer.javaTextNormalizer(java.text.Normalizer.Form.NFKC);
 * CharSequence normalized = nfkcNormalizer.apply("İstanbul");
 * }</pre>
 *
 * @see java.text.Normalizer
 * @see Form
 */
@FunctionalInterface
public interface Normalizer extends Function<CharSequence, CharSequence> {

    /**
     * A no-operation normalizer that returns the input text unchanged. This can be used as a
     * placeholder when normalization is optional or should be skipped.
     */
    Normalizer IDENTITY = text -> text;

    /**
     * Creates a new Normalizer that uses the specified {@link Form} to normalize text according to
     * Unicode normalization rules.
     *
     * <p>The supported normalization forms are:
     *
     * <ul>
     *   <li>{@link Form#NFD} - Canonical Decomposition
     *   <li>{@link Form#NFC} - Canonical Decomposition followed by Canonical Composition
     *   <li>{@link Form#NFKD} - Compatibility Decomposition
     *   <li>{@link Form#NFKC} - Compatibility Decomposition followed by Canonical Composition
     * </ul>
     *
     * @param form the Unicode normalization form to apply
     * @return a new Normalizer that applies the specified normalization form
     * @throws NullPointerException if form is null
     * @see java.text.Normalizer
     * @see Form
     */
    static Normalizer unicode(Form form) {
        return text -> java.text.Normalizer.normalize(text, form);
    }
}
