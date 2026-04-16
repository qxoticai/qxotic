package com.qxotic.toknroll;

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
 * <p>A {@code TokenizationModel} handles ordinary text chunks only. Special-token policy and
 * injection are handled separately (see {@link Specials}).
 *
 * <p>Strict model guarantees and limits:
 *
 * <ul>
 *   <li>{@code decode(encode(text)) == text} for ordinary text model paths.
 *   <li>{@code encode(decode(tokens)) == tokens} is not guaranteed in general.
 *   <li>{@link Tokenizer#decodeBytes(IntSequence)} is the authoritative non-lossy decode path.
 *   <li>Special-token injection (see {@link Specials}) is policy-driven and outside ordinary model
 *       round-trip guarantees.
 * </ul>
 *
 * <p>Note: a {@code TokenizationModel} is also a {@link Tokenizer} and can be used directly when
 * the input is already pre-split. {@link TokenizationPipeline} implements {@link Tokenizer} but not
 * {@code TokenizationModel}, so it cannot be nested as a model inside another pipeline.
 */
public interface TokenizationModel extends Tokenizer {}
