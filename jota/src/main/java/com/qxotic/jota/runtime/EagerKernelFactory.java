package com.qxotic.jota.runtime;

import java.util.Locale;

public final class EagerKernelFactory {

    private static final String MODE_PROPERTY = "jota.eager.kernels.mode";
    private static final String MODE_TRACE = "trace";
    private static final String MODE_BUNDLE = "bundle";

    private EagerKernelFactory() {}

    public static DeviceRuntime.EagerKernels create(DeviceRuntime runtime) {
        EagerKernelBundleLoader.bindManifestAliases(runtime);
        String mode = System.getProperty(MODE_PROPERTY, MODE_TRACE).trim().toLowerCase(Locale.ROOT);
        if (MODE_BUNDLE.equals(mode)) {
            return new LoadOnlyEagerKernels(runtime);
        }
        if (!MODE_TRACE.equals(mode)) {
            throw new IllegalArgumentException(
                    "Unsupported eager kernel mode: "
                            + mode
                            + " (expected '"
                            + MODE_TRACE
                            + "' or '"
                            + MODE_BUNDLE
                            + "')");
        }
        return new TracingEagerKernels();
    }
}
