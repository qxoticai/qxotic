package com.qxotic.jinfer.spi;

import com.qxotic.jinfer.FloatSequence;

/**
 * SPI hatch - plumbing between library modules (model ports, samplers, kernels), NOT end-user API.
 * A {@link FloatSequence} implementation backed by a heap array implements this so internals can
 * crack the potato: zero-copy access to the innards, mutation included (the sampler's
 * masking/scaling operates on the backing array directly - the read-only face is for everyone
 * else). Consumers MUST tolerate a potato that implements no hatch they know: fall back to {@link
 * FloatSequence#copyTo}.
 *
 * <p>Validity tracks the potato's own contract: the array of a working-buffer potato (logits) is
 * refilled by the producer's next call.
 */
public interface Heap extends FloatSequence {

    /** The backing array; {@code get(i)} lives at {@code array()[offset() + i]}. */
    float[] array();

    int offset();
}
