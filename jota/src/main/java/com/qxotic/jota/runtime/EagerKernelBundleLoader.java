package com.qxotic.jota.runtime;

import com.qxotic.jota.tensor.KernelCacheKey;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

final class EagerKernelBundleLoader {

    private EagerKernelBundleLoader() {}

    static void bindManifestAliases(DeviceRuntime runtime) {
        String resource =
                "/ai/qxotic/jota/kernels/eager/" + runtime.device().leafName() + ".properties";
        try (InputStream in = EagerKernelBundleLoader.class.getResourceAsStream(resource)) {
            if (in == null) {
                return;
            }
            Properties props = new Properties();
            props.load(in);
            for (String name : props.stringPropertyNames()) {
                String keyValue = props.getProperty(name);
                if (keyValue == null || keyValue.isBlank()) {
                    continue;
                }
                runtime.bindKernelName(name.trim(), KernelCacheKey.of(keyValue.trim()));
            }
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to load eager kernel manifest for device: " + runtime.device(), e);
        }
    }
}
