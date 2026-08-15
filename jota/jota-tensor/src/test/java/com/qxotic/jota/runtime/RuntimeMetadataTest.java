package com.qxotic.jota.runtime;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryDomain;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Test;

class RuntimeMetadataTest {

    @Test
    void mapBasedPropertiesAreSimpleAndImmutable() {
        var mutable = new LinkedHashMap<String, String>();
        mutable.put("device.name", "TestGPU");
        mutable.put("memory.global.bytes", "1024");

        Map<String, String> properties = Map.copyOf(mutable);
        mutable.put("compute.units", "64");

        assertEquals("TestGPU", properties.get("device.name"));
        assertEquals("1024", properties.get("memory.global.bytes"));
        assertFalse(properties.containsKey("compute.units"));
        assertThrows(UnsupportedOperationException.class, () -> properties.put("x", "y"));
    }

    @Test
    void setBasedCapabilitiesAreSimpleAndImmutable() {
        Set<String> capabilities = Set.of("fp32", "kernel.compilation", "atomic.32");

        assertTrue(capabilities.contains("fp32"));
        assertFalse(capabilities.contains("bf16"));
        assertThrows(UnsupportedOperationException.class, () -> capabilities.add("bf16"));
    }

    @Test
    void deviceRuntimeDefaultsExposeEmptyCollections() {
        DeviceRuntime runtime =
                new DeviceRuntime() {
                    @Override
                    public Device device() {
                        return null;
                    }

                    @Override
                    public MemoryDomain<?> memoryDomain() {
                        return null;
                    }

                    @Override
                    public ComputeEngine computeEngine() {
                        return null;
                    }

                    @Override
                    public Optional<KernelService> kernelService() {
                        return Optional.empty();
                    }
                };

        assertTrue(runtime.properties().isEmpty());
        assertTrue(runtime.capabilities().isEmpty());
    }
}
