package com.llm4j.api;

import java.util.function.ToIntFunction;

/**
 * A functional interface for sampling tokens from a language model's output distribution.
 * This interface extends ToIntFunction to transform a model state into a selected token ID
 * based on the model's output probabilities.
 * <p>
 * Implementations of this interface can provide different sampling strategies such as:
 * - Greedy sampling (selecting the highest probability token)
 * - Temperature-based sampling
 * - Top-k sampling
 * - Top-p (nucleus) sampling
 *
 * @param <State> The type representing the model's state containing output probabilities
 *                or logits from which to sample the next token
 */
@FunctionalInterface
public interface Sampler<State> extends ToIntFunction<State> {
}