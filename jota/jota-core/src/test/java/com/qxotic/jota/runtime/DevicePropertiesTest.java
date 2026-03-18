package com.qxotic.jota.runtime;

import static org.junit.jupiter.api.Assertions.*;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.OptionalLong;
import java.util.Set;
import org.junit.jupiter.api.Test;

class DevicePropertiesTest {

    @Test
    void emptyPropertiesHasNoKeys() {
        DeviceProperties props = DeviceProperties.EMPTY;
        assertFalse(props.has(DeviceProperties.DEVICE_NAME));
        assertTrue(props.asMap().isEmpty());
    }

    @Test
    void getStringReturnsValue() {
        var map = Map.<String, Object>of(DeviceProperties.DEVICE_NAME, "TestGPU");
        var props = new DeviceProperties(map);
        assertEquals("TestGPU", props.getString(DeviceProperties.DEVICE_NAME));
        assertEquals("TestGPU", props.name());
    }

    @Test
    void getLongReturnsValue() {
        var map = Map.<String, Object>of(DeviceProperties.GLOBAL_MEMORY_BYTES, 1024L);
        var props = new DeviceProperties(map);
        assertEquals(1024L, props.getLong(DeviceProperties.GLOBAL_MEMORY_BYTES));
        assertEquals(1024L, props.globalMemoryBytes());
    }

    @Test
    void getBooleanReturnsValue() {
        var map = Map.<String, Object>of("test.flag", true);
        var props = new DeviceProperties(map);
        assertTrue(props.getBoolean("test.flag"));
    }

    @Test
    void optionalLongReturnsEmptyForMissing() {
        var props = DeviceProperties.EMPTY;
        assertEquals(OptionalLong.empty(), props.optionalLong(DeviceProperties.COMPUTE_UNITS));
    }

    @Test
    void optionalLongReturnsValueWhenPresent() {
        var map = Map.<String, Object>of(DeviceProperties.COMPUTE_UNITS, 64L);
        var props = new DeviceProperties(map);
        assertEquals(OptionalLong.of(64L), props.optionalLong(DeviceProperties.COMPUTE_UNITS));
    }

    @Test
    void getMissingKeyThrows() {
        var props = DeviceProperties.EMPTY;
        assertThrows(IllegalArgumentException.class, () -> props.getString("missing"));
        assertThrows(IllegalArgumentException.class, () -> props.getLong("missing"));
        assertThrows(IllegalArgumentException.class, () -> props.getBoolean("missing"));
    }

    @Test
    void wrongTypeThrows() {
        var map = Map.<String, Object>of("key", "not-a-number");
        var props = new DeviceProperties(map);
        assertThrows(IllegalArgumentException.class, () -> props.getLong("key"));
        assertThrows(IllegalArgumentException.class, () -> props.getBoolean("key"));
    }

    @Test
    void toStringFormatsNicely() {
        assertEquals("DeviceProperties{}", DeviceProperties.EMPTY.toString());
        var map = Map.<String, Object>of(DeviceProperties.DEVICE_NAME, "GPU");
        var props = new DeviceProperties(map);
        assertTrue(props.toString().contains("device.name = GPU"));
    }

    @Test
    void mapIsImmutable() {
        var mutable = new LinkedHashMap<String, Object>();
        mutable.put("key", "value");
        var props = new DeviceProperties(mutable);
        mutable.put("key2", "value2");
        assertFalse(props.has("key2"));
        assertThrows(UnsupportedOperationException.class, () -> props.asMap().put("x", "y"));
    }

    @Test
    void emptyCapabilities() {
        var caps = DeviceCapabilities.EMPTY;
        assertFalse(caps.has(DeviceCapabilities.FP64));
        assertTrue(caps.asSet().isEmpty());
    }

    @Test
    void capabilitiesHasWorks() {
        var caps = new DeviceCapabilities(Set.of(DeviceCapabilities.FP32, DeviceCapabilities.FP64));
        assertTrue(caps.has(DeviceCapabilities.FP32));
        assertTrue(caps.has(DeviceCapabilities.FP64));
        assertFalse(caps.has(DeviceCapabilities.FP16));
    }

    @Test
    void capabilitiesSetIsImmutable() {
        var caps = new DeviceCapabilities(Set.of(DeviceCapabilities.FP32));
        assertThrows(UnsupportedOperationException.class, () -> caps.asSet().add("x"));
    }

    @Test
    void capabilitiesToString() {
        assertEquals("DeviceCapabilities{}", DeviceCapabilities.EMPTY.toString());
        var caps = new DeviceCapabilities(Set.of(DeviceCapabilities.FP32));
        assertTrue(caps.toString().contains("fp32"));
    }

    @Test
    void deviceRuntimeDefaults() {
        DeviceRuntime runtime =
                new DeviceRuntime() {
                    @Override
                    public com.qxotic.jota.Device device() {
                        return null;
                    }

                    @Override
                    public com.qxotic.jota.memory.MemoryDomain<?> memoryDomain() {
                        return null;
                    }

                    @Override
                    public ComputeEngine computeEngine() {
                        return null;
                    }

                    @Override
                    public java.util.Optional<KernelService> kernelService() {
                        return java.util.Optional.empty();
                    }
                };
        assertSame(DeviceProperties.EMPTY, runtime.properties());
        assertSame(DeviceCapabilities.EMPTY, runtime.capabilities());
    }
}
