package ai.qxotic.jota.memory;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

class MemoryOperationsTest {

    @ParameterizedTest
    @MethodSource("ai.qxotic.jota.memory.AbstractMemoryTest#domainsSupportingF32")
    <B> void copyShouldTransferDataBetweenBuffers(MemoryDomain<B> domain) {
        MemoryAllocator<B> memoryAllocator = domain.memoryAllocator();
        MemoryOperations<B> memoryOperations = domain.memoryOperations();
        MemoryAccess<B> memoryAccess = domain.directAccess();

        Memory<B> src = memoryAllocator.allocateMemory(8);
        Memory<B> dst = memoryAllocator.allocateMemory(8);

        // Initialize source
        float value = (float) Math.PI;
        memoryAccess.writeFloat(src, 0, value);

        // Perform copy
        memoryOperations.copy(src, 0, dst, 0, 4);

        // Verify
        assertEquals(value, memoryAccess.readFloat(dst, 0));
    }
}
