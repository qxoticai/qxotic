package com.qxotic.toknroll.advanced;

import com.qxotic.toknroll.Tokenizer;

/**
 * Marker interface for pure tokenization models (chunk encoders).
 *
 * <p>A {@code TokenizationModel} encodes a single pre-split, pre-normalised chunk of text. Passing
 * unsplit text produces incorrect results. Splitting is the responsibility of the caller (typically
 * {@link TokenizationPipeline}).
 *
 * <p>Implement this interface to plug in custom models (unigram, custom BPE variants, etc.) into a
 * {@link TokenizationPipeline}.
 *
 * <p>Note: a {@code TokenizationModel} is also a {@link Tokenizer} and can be used directly when
 * the input is already pre-split. {@link TokenizationPipeline} implements {@link Tokenizer} but not
 * {@code TokenizationModel}, so it cannot be nested as a model inside another pipeline.
 */
public interface TokenizationModel extends Tokenizer {}
