package com.qxotic.jota.runtime.panama;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.nativeimpl.NativeMemoryFactory;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

public final class PanamaRuntimeProvider extends DeviceRuntimeProvider {

    @Override
    public DeviceType deviceType() {
        return DeviceType.PANAMA;
    }

    @Override
    public int priority() {
        return 1000;
    }

    @Override
    public RuntimeProbe probe() {
        try {
            NativeMemoryFactory.createDomain();
            return RuntimeProbe.available("Panama runtime available");
        } catch (Throwable t) {
            return RuntimeProbe.error("Panama runtime failed to initialize", t);
        }
    }

    @Override
    protected DeviceRuntime createForDevice(Device device) {
        return new PanamaDeviceRuntime();
    }

    @Override
    public Map<String, String> properties(int deviceIndex) {
        Runtime rt = Runtime.getRuntime();
        var props = new LinkedHashMap<String, String>();
        props.put("device.name", "JVM (" + System.getProperty("java.vm.name") + ")");
        props.put("device.vendor", System.getProperty("java.vm.vendor"));
        props.put("device.architecture", System.getProperty("os.arch"));
        props.put("device.driver.version", System.getProperty("java.runtime.version"));
        props.put("memory.global.bytes", Long.toString(rt.maxMemory()));
        props.put("compute.units", Integer.toString(rt.availableProcessors()));
        props.put("device.kind", "cpu");
        return Map.copyOf(props);
    }

    @Override
    public Set<String> capabilities(int deviceIndex) {
        return Set.of(
                "fp32", "fp64", "kernel.compilation", "native.runtime", "unified.memory", "cpu");
    }
}
