package com.llm4j.tokenizers;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

/**
 * A functional interface for text splitting operations used in tokenization pipelines.
 * TextSplitter breaks input text into smaller chunks before final tokenization,
 * serving as an initial preprocessing step.
 *
 * <p>Common splitting operations include:
 * <ul>
 *   <li>Breaking text on whitespace boundaries</li>
 *   <li>Separating punctuation from words</li>
 *   <li>Splitting text into sentences</li>
 *   <li>Isolating special tokens (URLs, emails, etc.)</li>
 * </ul>
 *
 * <p>Example usage:
 * <pre>{@code
 * // Simple whitespace splitter
 * TextSplitter whitespace = text -> Arrays.asList(text.toString().split("\\s+"));
 *
 * // Using in a tokenization pipeline
 * text -> normalizer.apply(text)         // First normalize
 *      -> splitter.apply(text)           // Then split into chunks
 *      -> chunks.stream().map(tokenizer) // Finally tokenize each chunk
 * }</pre>
 *
 * @see Normalizer
 * @see Tokenizer
 */
@FunctionalInterface
public interface TextSplitter extends Function<CharSequence, List<CharSequence>> {
    /**
     * A splitter that returns the input text as a single chunk.
     * This can be used as a placeholder when splitting is optional or should be skipped.
     */
    TextSplitter IDENTITY = List::of;

    /**
     * Splits the input text into a list of chunks according to this splitter's rules.
     *
     * @param text the text to split
     * @return a list of character sequences representing the split chunks
     * @throws NullPointerException if text is null
     */
    @Override
    List<CharSequence> apply(CharSequence text);

    /**
     * Creates a new TextSplitter that sequentially applies multiple splitters.
     * Each splitter in the sequence operates on the chunks produced by the previous splitter.
     *
     * <p>Example usage:
     * <pre>{@code
     * TextSplitter combined = TextSplitter.compose(
     *     whitespaceSpitter,
     *     punctuationSplitter
     * );
     * }</pre>
     *
     * @param splitters the sequence of splitters to apply
     * @return a new TextSplitter that combines the given splitters
     * @throws NullPointerException if splitters is null or contains null elements
     */
    static TextSplitter compose(TextSplitter... splitters) {
        return text -> {
            List<CharSequence> current = new ArrayList<>();
            current.add(text);

            for (TextSplitter splitter : splitters) {
                List<CharSequence> next = new ArrayList<>();
                for (CharSequence chunk : current) {
                    next.addAll(splitter.apply(chunk));
                }
                current = next;
            }

            return current;
        };
    }
}