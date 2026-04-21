package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

public final class OpenClRuntimeProvider extends DeviceRuntimeProvider {

    @Override
    public DeviceType deviceType() {
        return DeviceType.OPENCL;
    }

    @Override
    public RuntimeProbe probe() {
        if (!OpenClRuntime.isAvailable()) {
            return RuntimeProbe.missingSoftware(
                    "OpenCL JNI runtime library is not available",
                    OpenClRuntime.availabilityDetails());
        }
        try {
            int count = OpenClRuntime.deviceCount();
            if (count <= 0) {
                return RuntimeProbe.unsupportedHardware(
                        "OpenCL runtime is present but no compatible CPU/GPU device was detected",
                        "Install/enable an OpenCL ICD and verify at least one OpenCL CPU/GPU device"
                                + " is visible");
            }
            return RuntimeProbe.available(
                    "OpenCL runtime available with "
                            + count
                            + " device(s); selected "
                            + OpenClRuntime.selectedDeviceType()
                            + " on platform '"
                            + OpenClRuntime.selectedPlatformName()
                            + "'"
                            + " device '"
                            + OpenClRuntime.deviceName()
                            + "' ("
                            + OpenClRuntime.selectionPropertiesSummary()
                            + ")");
        } catch (RuntimeException e) {
            return RuntimeProbe.misconfigured(
                    "OpenCL runtime is loaded but device probing failed",
                    "Check OpenCL driver/ICD installation and OpenCL selection properties: "
                            + OpenClRuntime.selectionPropertiesSummary()
                            + " | last init error: "
                            + OpenClRuntime.initFailureReason(),
                    e);
        }
    }

    @Override
    protected DeviceRuntime createForDevice(Device device) {
        return new OpenClDeviceRuntime(device);
    }

    @Override
    public Map<String, String> properties(int deviceIndex) {
        if (!OpenClRuntime.isAvailable()) {
            return Map.of();
        }
        var props = new LinkedHashMap<String, String>();
        props.put("device.name", OpenClRuntime.deviceName());
        props.put("device.vendor", OpenClRuntime.deviceVendor());
        props.put("device.kind", OpenClRuntime.selectedDeviceType().toLowerCase());
        props.put("opencl.platform", OpenClRuntime.selectedPlatformName());
        return Map.copyOf(props);
    }

    @Override
    public Set<String> capabilities(int deviceIndex) {
        if (!OpenClRuntime.isAvailable()) {
            return Set.of();
        }
        String kind = OpenClRuntime.selectedDeviceType().toLowerCase();
        return Set.of(kind, "fp32", "kernel.compilation");
    }
}
