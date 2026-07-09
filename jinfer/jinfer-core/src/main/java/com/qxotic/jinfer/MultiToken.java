package com.qxotic.jinfer;

import java.util.OptionalInt;

/**
 * Optional capability: multi-token prediction (speculative / MTP heads). A model implements this
 * iff its <em>architecture</em> supports MTP — {@code instanceof MultiToken} is that test. Whether
 * MTP is usable right now is a separate, runtime fact: {@link #depth()} is <em>empty</em> when
 * supported-but-not-loaded (no draft heads in this checkpoint) and present (≥1) when loaded. The
 * empty gate is un-ignorable, unlike a sentinel {@code 0} — you cannot draft without unwrapping it.
 * When present, the generation loop drafts {@code depth()} tokens past the sampled one, then
 * verifies.
 */
public interface MultiToken<S extends RuntimeState> {
    OptionalInt
            depth(); // empty ⇒ supported but no draft loaded; present ⇒ number of draft heads (≥ 1)

    FloatTensor logits(S state, int output, int head);
}
