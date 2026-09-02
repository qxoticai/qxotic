package com.qxotic.toknroll;

/**
 * Marker interface for token classification schemes. Implementations (typically enums) categorize
 * tokens by role - normal text, control markers, raw bytes - so vocabularies and loaders can attach
 * per-token behavior without a fixed category set.
 *
 * @see StandardTokenType
 */
public interface TokenType {}
