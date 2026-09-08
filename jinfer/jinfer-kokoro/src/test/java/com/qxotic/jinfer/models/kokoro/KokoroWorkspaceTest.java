package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class KokoroWorkspaceTest {

    @Test
    void reusesTheHighWaterBuffersAfterRewind() {
        try (Arena arena = Arena.ofShared()) {
            var workspace = new KokoroWorkspace(MemoryAllocators.ofArena(arena));
            Memory<MemorySegment> first = workspace.allocateMemory(64, 64);
            Memory<MemorySegment> second = workspace.allocateMemory(128, 64);

            workspace.rewind();
            assertSame(first, workspace.allocateMemory(32, 64));
            assertSame(second, workspace.allocateMemory(128, 64));

            workspace.rewind();
            Memory<MemorySegment> grown = workspace.allocateMemory(65, 64);
            assertEquals(65, grown.byteSize());
            workspace.rewind();
            assertSame(grown, workspace.allocateMemory(65, 64));
        }
    }
}
