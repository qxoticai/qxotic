package com.qxotic.jinfer.x.boundary;

/** Runtime defaults owned by the model boundary. */
final class RuntimeFlags {

    /** Default scratch width for {@code newState}: prefill batches up to this many tokens. */
    public static final int BATCH_CAPACITY = Integer.getInteger("jinfer.batchCapacity", 512);

    private RuntimeFlags() {}
}
