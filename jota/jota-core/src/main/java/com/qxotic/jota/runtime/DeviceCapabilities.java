package com.qxotic.jota.runtime;

import java.util.Set;
import java.util.TreeSet;

public final class DeviceCapabilities {

    public static final DeviceCapabilities EMPTY = new DeviceCapabilities(Set.of());

    private final Set<String> capabilities;

    public DeviceCapabilities(Set<String> capabilities) {
        this.capabilities = Set.copyOf(capabilities);
    }

    public boolean has(String capability) {
        return capabilities.contains(capability);
    }

    public Set<String> asSet() {
        return capabilities;
    }

    @Override
    public String toString() {
        if (capabilities.isEmpty()) {
            return "DeviceCapabilities{}";
        }
        return "DeviceCapabilities" + new TreeSet<>(capabilities);
    }

    // --- Standard capability keys ---

    // Data type support
    public static final String FP16 = "fp16";
    public static final String BF16 = "bf16";
    public static final String FP32 = "fp32";
    public static final String FP64 = "fp64";
    public static final String TF32 = "tf32";
    public static final String INT8 = "int8";

    // Memory
    public static final String UNIFIED_MEMORY = "unified.memory";
    public static final String ECC_MEMORY = "ecc.memory";
    public static final String HOST_MAPPED_MEMORY = "host.mapped.memory";

    // Execution
    public static final String KERNEL_COMPILATION = "kernel.compilation";
    public static final String NATIVE_RUNTIME = "native.runtime";
    public static final String CONCURRENT_KERNELS = "concurrent.kernels";
    public static final String COOPERATIVE_GROUPS = "cooperative.groups";

    // Atomics
    public static final String ATOMIC_32 = "atomic.32";
    public static final String ATOMIC_64 = "atomic.64";
}
