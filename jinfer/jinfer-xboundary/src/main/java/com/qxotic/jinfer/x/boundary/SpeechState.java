package com.qxotic.jinfer.x.boundary;

/**
 * A speech model's scratch, and the one thing a caller must do with it: close it.
 *
 * <p>Narrows {@link AutoCloseable} to a {@code close()} that throws nothing - freeing scratch
 * cannot fail, and the checked exception would otherwise reach every caller of a wildcarded {@link
 * SpeechModel}, including anyone loading through architecture dispatch.
 *
 * <p>Close is idempotent, and it must come AFTER the last {@code speak} using this state returns:
 * the kernels read raw addresses, so a live read from a closed arena is a crash, not an exception.
 */
public interface SpeechState extends AutoCloseable {

    @Override
    void close();
}
