package ai.qxotic.jota.hip;

import static org.junit.jupiter.api.Assertions.assertEquals;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Indexing;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.impl.ContextFactory;
import ai.qxotic.jota.memory.impl.MemoryFactory;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

class HipMemoryTest {

    @Test
    void copyFromAndToNative() {
        Assumptions.assumeTrue(HipRuntime.isAvailable());

        MemoryContext<MemorySegment> host = ContextFactory.ofMemorySegment();
        HipMemoryContext device = HipMemoryContext.instance();

        byte[] data = new byte[] {1, 2, 3, 4};
        MemoryView<MemorySegment> hostView =
                MemoryView.of(
                        MemoryFactory.ofMemorySegment(MemorySegment.ofArray(data)),
                        DataType.I8,
                        Layout.rowMajor(Shape.flat(data.length)));

        MemoryView<HipDevicePtr> deviceView =
                MemoryView.of(
                        device.memoryAllocator().allocateMemory(DataType.I8, data.length),
                        DataType.I8,
                        Layout.rowMajor(Shape.flat(data.length)));

        device.memoryOperations()
                .copyFromNative(
                        hostView.memory(),
                        hostView.byteOffset(),
                        deviceView.memory(),
                        0,
                        data.length);

        MemorySegment outSegment = MemorySegment.ofArray(new byte[data.length]);
        MemoryView<MemorySegment> outView =
                MemoryView.of(
                        MemoryFactory.ofMemorySegment(outSegment),
                        DataType.I8,
                        Layout.rowMajor(Shape.flat(data.length)));

        device.memoryOperations()
                .copyToNative(
                        deviceView.memory(),
                        0,
                        outView.memory(),
                        outView.byteOffset(),
                        data.length);

        MemoryAccess<MemorySegment> access = host.memoryAccess();
        for (int i = 0; i < data.length; i++) {
            long offset = Indexing.linearToOffset(outView, i);
            assertEquals(data[i], access.readByte(outView.memory(), offset));
        }
    }
}
