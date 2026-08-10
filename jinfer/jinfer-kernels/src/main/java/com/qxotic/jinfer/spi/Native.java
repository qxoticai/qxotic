package com.qxotic.jinfer.spi;

import com.qxotic.jinfer.FloatSequence;
import com.qxotic.jinfer.FloatTensor;

/**
 * SPI hatch (jinfer-kernels) - plumbing between library modules, NOT end-user API. A {@link
 * FloatSequence} backed by a native {@link FloatTensor} (a model's logits buffer, an embedder's
 * chunk scratch) implements this so internals can crack the potato and keep working in the tensor
 * world - zero copies, mutation included. Consumers MUST tolerate potatoes with no hatch they know:
 * fall back to {@link FloatSequence#copyTo}. Validity tracks the potato's own contract - a
 * working-buffer potato's tensor is rewritten by the producer's next call, and the arena laws of
 * the owning state apply (a stale potato after the state's arena closed is a crash, exactly like
 * borrowing the tensor itself).
 */
public interface Native extends FloatSequence {

    /** The backing tensor; {@code get(i)} lives at {@code tensor().getFloat(offset() + i)}. */
    FloatTensor tensor();

    long offset();
}
