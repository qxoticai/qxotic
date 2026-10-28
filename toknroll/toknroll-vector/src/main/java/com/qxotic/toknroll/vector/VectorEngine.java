package com.qxotic.toknroll.vector;

import com.qxotic.toknroll.spi.TiktokenEngine;
import com.qxotic.toknroll.spi.TiktokenTables;

/**
 * The tiktoken family encoded with the Java Vector API. Found by core through {@code ServiceLoader}
 * when the JVM runs with {@code --add-modules jdk.incubator.vector}.
 */
public final class VectorEngine implements TiktokenEngine {

    @Override
    public Encoder encoder(TiktokenTables model, int rules) {
        BitSplit split = BitSplit.forRules(rules);
        return split == null ? null : new FastEncoder(model, split);
    }
}
