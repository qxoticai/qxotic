package com.qxotic.jota.runtime.c;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

final class CRuntimeMetadata {

    private static final Set<String> CAPABILITIES =
            Set.of("cpu", "fp32", "fp64", "kernel.compilation", "native.runtime", "unified.memory");

    private CRuntimeMetadata() {}

    static Map<String, String> properties() {
        Runtime rt = Runtime.getRuntime();
        var props = new LinkedHashMap<String, String>();
        props.put("device.name", "C Host");
        props.put("device.vendor", System.getProperty("os.name"));
        props.put("device.architecture", System.getProperty("os.arch"));
        props.put("memory.global.bytes", Long.toString(rt.maxMemory()));
        props.put("compute.units", Integer.toString(rt.availableProcessors()));
        props.put("device.kind", "cpu");
        return Map.copyOf(props);
    }

    static Set<String> capabilities() {
        return CAPABILITIES;
    }
}
