/**
 * Advanced tokenizer composition APIs.
 *
 * <p>This package is intended for power users and adapter modules. APIs in this package may change
 * more frequently than the stable {@code com.qxotic.tokenizers} surface.
 *
 * <p>Default behavior should stay strict and non-lossy (identity normalization and deterministic
 * splitting). Any lossy normalization step (for example lowercasing or accent stripping) is
 * explicit and opt-in.
 */
package com.qxotic.tokenizers.advanced;
