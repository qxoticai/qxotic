package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.DeviceType;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.spi.DeviceRuntimeProvider;
import com.qxotic.jota.runtime.spi.RuntimeProbe;

public final class OpenClRuntimeProvider implements DeviceRuntimeProvider {

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
    public DeviceRuntime create(int deviceIndex) {
        return new OpenClDeviceRuntime();
    }
}
