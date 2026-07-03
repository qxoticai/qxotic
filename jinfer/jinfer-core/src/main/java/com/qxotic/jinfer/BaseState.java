package com.qxotic.jinfer;

/** The shared cursor of every model State: how far the sequence has been ingested and what the
 *  last ingest retained. Owns the {@link #advance}/{@link #resumeAt} lifecycle so each model's
 *  State carries no cursor boilerplate. Fields are public for the model's own forward code
 *  (hot-path reads like {@code s.position} across packages); mutation belongs to the two
 *  lifecycle methods only. */
public abstract class BaseState implements RuntimeState {

    public int position;       // tokens ingested so far
    public int outputCount;    // hidden states the last ingest retained (1 after LAST, n after ALL)
    public int lastChunkLen;   // rows of the last ingested batch

    @Override
    public final int position() {
        return position;
    }

    @Override
    public final int outputCount() {
        return outputCount;
    }

    @Override
    public final void advance(int rows, Batch.Outputs outputs) {
        lastChunkLen = rows;
        outputCount = outputs == Batch.Outputs.ALL ? rows : 1;
        position += rows;
    }

    @Override
    public final void resumeAt(int p) {
        position = p;
        lastChunkLen = 0;
        outputCount = 0;
    }
}
