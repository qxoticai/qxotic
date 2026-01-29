package ai.qxotic.jota.ir.interpreter;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.ir.irt.IRTNode;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.ScopedMemoryAllocatorArena;
import ai.qxotic.jota.panama.PanamaFactory;
import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class IRTEvalContext implements AutoCloseable {

    private final ScopedMemoryAllocatorArena<MemorySegment> arena;
    private final Map<Integer, MemoryView<MemorySegment>> inputMap;
    private final Map<IRTNode, MemoryView<MemorySegment>> resultCache;
    private final MemoryAccess<MemorySegment> memAccess;

    private IRTEvalContext(
            ScopedMemoryAllocatorArena<MemorySegment> arena,
            Map<Integer, MemoryView<MemorySegment>> inputMap,
            Map<IRTNode, MemoryView<MemorySegment>> resultCache,
            MemoryAccess<MemorySegment> memAccess) {
        this.arena = arena;
        this.inputMap = inputMap;
        this.resultCache = resultCache;
        this.memAccess = memAccess;
    }

    @SuppressWarnings("unchecked")
    public static IRTEvalContext create(List<MemoryView<?>> inputs, MemoryContext<?> context) {
        Map<Integer, MemoryView<MemorySegment>> inputMap = new HashMap<>();
        for (int i = 0; i < inputs.size(); i++) {
            inputMap.put(i, (MemoryView<MemorySegment>) inputs.get(i));
        }

        MemoryAccess<MemorySegment> memAccess =
                (MemoryAccess<MemorySegment>) context.memoryAccess();
        ScopedMemoryAllocatorArena<MemorySegment> arena = PanamaFactory.createArena();

        return new IRTEvalContext(arena, inputMap, new HashMap<>(), memAccess);
    }

    public MemoryView<MemorySegment> getInput(int tensorId) {
        MemoryView<MemorySegment> input = inputMap.get(tensorId);
        if (input == null) {
            throw new IllegalArgumentException("Unknown tensor input id: " + tensorId);
        }
        return input;
    }

    public MemoryView<MemorySegment> evaluate(IRTNode node) {
        MemoryView<MemorySegment> result = resultCache.get(node);
        if (result == null) {
            result = node.accept(new IRTEvalVisitor(this));
            resultCache.put(node, result);
        }
        return result;
    }

    public MemoryView<MemorySegment> allocateOutput(DataType dtype, Layout layout) {
        long size = layout.shape().size();
        Memory<MemorySegment> memory = arena.allocateMemory(dtype, size);
        return MemoryView.of(memory, 0, dtype, layout);
    }

    public MemoryView<MemorySegment> allocateTemporary(DataType dtype, Layout layout) {
        return allocateOutput(dtype, layout);
    }

    public MemoryAccess<MemorySegment> getMemoryAccess() {
        return memAccess;
    }

    @Override
    public void close() {
        arena.close();
    }
}
