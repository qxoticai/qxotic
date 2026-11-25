package com.qxotic.jota.stablehlo.ir;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/** A region: an ordered list of blocks (StableHLO regions have at most one block). */
public final class Region {

    private final List<Block> blocks = new ArrayList<>();

    public Region() {}

    public static Region of(Block block) {
        Region r = new Region();
        r.addBlock(block);
        return r;
    }

    public void addBlock(Block block) {
        blocks.add(block);
    }

    public List<Block> blocks() {
        return Collections.unmodifiableList(blocks);
    }

    public boolean isEmpty() {
        return blocks.isEmpty();
    }

    /** The single block of a StableHLO region. */
    public Block block() {
        if (blocks.size() != 1) {
            throw new IllegalStateException("region has " + blocks.size() + " blocks");
        }
        return blocks.getFirst();
    }
}
