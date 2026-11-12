package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.spi.TiktokenEngine;
import com.qxotic.toknroll.spi.TiktokenTables;
import java.util.concurrent.atomic.AtomicInteger;

/** A test engine that agrees with nothing, or runs out of memory while armed. */
public final class ArmedEngine implements TiktokenEngine {
    static volatile boolean armed;
    static final AtomicInteger asked = new AtomicInteger();

    @Override
    public Encoder encoder(TiktokenTables model, int rules) {
        asked.incrementAndGet();
        if (armed) throw new OutOfMemoryError("armed");
        return null;
    }
}
