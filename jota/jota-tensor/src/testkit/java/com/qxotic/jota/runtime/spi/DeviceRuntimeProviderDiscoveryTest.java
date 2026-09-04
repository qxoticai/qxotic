package com.qxotic.jota.runtime.spi;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.AvailableDevice;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.KernelService;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;

class DeviceRuntimeProviderDiscoveryTest {

    @Test
    void availableDevicesUsesProviderMetadataForEachIndex() {
        DeviceRuntimeProvider provider = new TestProvider(RuntimeProbe.available("ok"), 2);

        List<AvailableDevice> devices = provider.availableDevices();

        assertEquals(2, devices.size());
        assertEquals(DeviceType.fromId("test-provider").deviceIndex(0), devices.get(0).device());
        assertEquals(DeviceType.fromId("test-provider").deviceIndex(1), devices.get(1).device());
        assertEquals("0", devices.get(0).properties().get("index"));
        assertEquals("1", devices.get(1).properties().get("index"));
        assertTrue(devices.get(0).capabilities().contains("gpu"));
        assertTrue(devices.get(1).capabilities().contains("index:1"));
        assertThrows(UnsupportedOperationException.class, () -> devices.add(devices.get(0)));
    }

    @Test
    void availableDevicesReturnsEmptyWhenProbeUnavailable() {
        DeviceRuntimeProvider provider =
                new TestProvider(RuntimeProbe.missingSoftware("missing", "install it"), 2);

        assertTrue(provider.availableDevices().isEmpty());
    }

    @Test
    void createDeviceRejectsMismatchedRuntimeId() {
        DeviceRuntimeProvider provider = new TestProvider(RuntimeProbe.available("ok"), 1);
        Device wrong = DeviceType.fromId("other-runtime").deviceIndex(0);

        IllegalArgumentException error =
                assertThrows(IllegalArgumentException.class, () -> provider.create(wrong));
        assertTrue(error.getMessage().contains("cannot create runtime"));
    }

    @Test
    void createDeviceDelegatesRequestedIndex() {
        AtomicLong capturedIndex = new AtomicLong(-1);
        DeviceRuntimeProvider provider =
                new TestProvider(RuntimeProbe.available("ok"), 1) {
                    @Override
                    protected DeviceRuntime createForDevice(Device device) {
                        capturedIndex.set(device.index());
                        return new StubRuntime(device);
                    }
                };

        Device requested = DeviceType.fromId("test-provider").deviceIndex(7);
        DeviceRuntime runtime = provider.create(requested);

        assertEquals(7L, capturedIndex.get());
        assertEquals(requested, runtime.device());
    }

    @Test
    void createDeviceRejectsRuntimeWithWrongRuntimeId() {
        DeviceRuntimeProvider provider =
                new TestProvider(RuntimeProbe.available("ok"), 1) {
                    @Override
                    protected DeviceRuntime createForDevice(Device device) {
                        return new StubRuntime(
                                DeviceType.fromId("wrong")
                                        .deviceIndex(Math.toIntExact(device.index())));
                    }
                };

        Device requested = DeviceType.fromId("test-provider").deviceIndex(0);
        IllegalStateException error =
                assertThrows(IllegalStateException.class, () -> provider.create(requested));
        assertTrue(error.getMessage().contains("returned runtime bound"));
    }

    @Test
    void providerToStringUsesCompactFormat() {
        DeviceRuntimeProvider provider = new TestProvider(RuntimeProbe.available("ok"), 1);
        assertEquals(
                "DeviceRuntimeProvider{deviceType=test-provider, priority=0}", provider.toString());
    }

    private static class TestProvider extends DeviceRuntimeProvider {
        private final RuntimeProbe probe;
        private final int count;

        private TestProvider(RuntimeProbe probe, int count) {
            this.probe = probe;
            this.count = count;
        }

        @Override
        public DeviceType deviceType() {
            return DeviceType.fromId("test-provider");
        }

        @Override
        public RuntimeProbe probe() {
            return probe;
        }

        @Override
        public int deviceCount() {
            return count;
        }

        @Override
        public Map<String, String> properties(int deviceIndex) {
            var properties = new LinkedHashMap<String, String>();
            properties.put("index", Integer.toString(deviceIndex));
            return properties;
        }

        @Override
        public Set<String> capabilities(int deviceIndex) {
            var capabilities = new LinkedHashSet<String>();
            capabilities.add("gpu");
            capabilities.add("index:" + deviceIndex);
            return capabilities;
        }

        @Override
        protected DeviceRuntime createForDevice(Device device) {
            throw new UnsupportedOperationException("Not needed for discovery test");
        }
    }

    private record StubRuntime(Device device) implements DeviceRuntime {
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
    }
}
