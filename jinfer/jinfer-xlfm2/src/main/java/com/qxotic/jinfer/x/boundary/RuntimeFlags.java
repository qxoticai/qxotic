package com.qxotic.jinfer.x.boundary;

/** The boundary cone's slice of jinfer's runtime flags (ported from jinfer-core RuntimeFlags). */
final class RuntimeFlags {

    /** Default scratch width for {@code newState}: prefill batches up to this many tokens. */
    public static final int BATCH_CAPACITY = Integer.getInteger("jinfer.batchCapacity", 512);

    private RuntimeFlags() {}
}
