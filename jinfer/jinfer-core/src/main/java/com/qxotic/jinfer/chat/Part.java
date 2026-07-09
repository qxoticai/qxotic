package com.qxotic.jinfer.chat;

import com.qxotic.jinfer.Media;

/**
 * One piece of a message's content: text, or a decoded media payload. The sealed {@link Media}
 * union carries the modality (image / audio / video / future), so adding a modality never touches
 * this interface or {@link ChatTemplate}.
 */
public sealed interface Part {

    /**
     * Untrusted conversation text. Templates MUST tokenize it plainly (no special-token
     * recognition), so content can never mint control tokens.
     */
    record Text(String text) implements Part {
        public Text {
            if (text == null) throw new IllegalArgumentException("null text");
        }
    }

    /**
     * A decoded media payload, positioned structurally (by its place in the part list, never by
     * markers parsed out of text).
     */
    record Blob(Media media) implements Part {
        public Blob {
            if (media == null) throw new IllegalArgumentException("null media");
        }
    }
}
