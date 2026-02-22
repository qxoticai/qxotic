package com.qxotic.jota.ir.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.memory.*;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.panama.PanamaFactory;
import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class TIREvalContext implements AutoCloseable {

    private final ScopedMemoryAllocatorArena<MemorySegment> arena;
    private final Map<Integer, MemoryView<MemorySegment>> inputMap;
    private final Map<TIRNode, MemoryView<MemorySegment>> resultCache;
    private final MemoryAccess<MemorySegment> memAccess;

    private TIREvalContext(
            ScopedMemoryAllocatorArena<MemorySegment> arena,
            Map<Integer, MemoryView<MemorySegment>> inputMap,
            Map<TIRNode, MemoryView<MemorySegment>> resultCache,
            MemoryAccess<MemorySegment> memAccess) {
        this.arena = arena;
        this.inputMap = inputMap;
        this.resultCache = resultCache;
        this.memAccess = memAccess;
    }

    @SuppressWarnings("unchecked")
    public static TIREvalContext create(List<MemoryView<?>> inputs, MemoryDomain<?> memoryDomain) {
        Map<Integer, MemoryView<MemorySegment>> inputMap = new HashMap<>();
        for (int i = 0; i < inputs.size(); i++) {
            inputMap.put(i, (MemoryView<MemorySegment>) inputs.get(i));
        }

        MemoryAccess<MemorySegment> memAccess =
                (MemoryAccess<MemorySegment>) memoryDomain.directAccess();
        ScopedMemoryAllocatorArena<MemorySegment> arena = PanamaFactory.createArena();

        return new TIREvalContext(arena, inputMap, new HashMap<>(), memAccess);
    }

    public MemoryView<MemorySegment> getInput(int tensorId) {
        MemoryView<MemorySegment> input = inputMap.get(tensorId);
        if (input == null) {
            throw new IllegalArgumentException("Unknown tensor input id: " + tensorId);
        }
        return input;
    }

    public MemoryView<MemorySegment> evaluate(TIRNode node) {
        MemoryView<MemorySegment> result = resultCache.get(node);
        if (result == null) {
            result = node.accept(new TIREvalVisitor(this));
            resultCache.put(node, result);
        }
        return result;
    }

    public MemoryView<MemorySegment> allocateOutput(DataType dataType, Layout layout) {
        long size = layout.shape().size();
        Memory<MemorySegment> memory = arena.allocateMemory(dataType, size);
        return MemoryView.of(memory, 0, dataType, layout);
    }

    public MemoryView<MemorySegment> allocateTemporary(DataType dataType, Layout layout) {
        return allocateOutput(dataType, layout);
    }

    public MemoryAccess<MemorySegment> getMemoryAccess() {
        return memAccess;
    }

    @Override
    public void close() {
        arena.close();
    }
}
