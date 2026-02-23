package com.qxotic.tokenizers.advanced;

import com.qxotic.tokenizers.IntSequence;

@FunctionalInterface
public interface Encoder {

    /** Encodes one splitter chunk into token IDs. */
    IntSequence encode(CharSequence chunk);
}
