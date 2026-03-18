package com.qxotic.jota.runtime.opencl;

public final class OpenClDeviceListCli {

    private OpenClDeviceListCli() {}

    public static void main(String[] args) {
        System.out.println(
                "OpenCL selection properties: " + OpenClRuntime.selectionPropertiesSummary());
        if (!OpenClRuntime.isAvailable()) {
            System.err.println(
                    "OpenCL runtime unavailable: " + OpenClRuntime.availabilityDetails());
            System.exit(1);
            return;
        }

        try {
            System.out.println("Available OpenCL devices:");
            System.out.println(OpenClRuntime.listDevices());
            System.out.println("Selected platform: " + OpenClRuntime.selectedPlatformName());
            System.out.println("Selected device type: " + OpenClRuntime.selectedDeviceType());
            System.out.println("Selected device name: " + OpenClRuntime.deviceName());
        } catch (RuntimeException e) {
            System.err.println("Failed to query OpenCL devices: " + e.getMessage());
            System.err.println("Init failure reason: " + OpenClRuntime.initFailureReason());
            System.exit(2);
        }
    }
}
