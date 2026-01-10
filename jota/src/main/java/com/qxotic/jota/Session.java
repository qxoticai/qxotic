package com.qxotic.jota;

import java.util.HashMap;
import java.util.Map;

public final class Session { // implements AutoCloseable {
//
//    private static final ThreadLocal<Session> CURRENT = new ThreadLocal<>();
//
//    private final Map<Device, Backend> backends;
//    private final Device defaultDevice;
//
//    private Session(Map<Device, Backend> backends, Device defaultDevice) {
//        this.backends = Map.copyOf(backends);
//        this.defaultDevice = defaultDevice;
//    }
//
//    public Device defaultDevice() {
//        return defaultDevice;
//    }
//
//    public Backend backend(Device device) {
//        Backend backend = backends.get(device);
//        if (backend == null) {
//            throw new IllegalArgumentException("No backend registered for device: " + device);
//        }
//        return backend;
//    }
//
//    public Backend defaultBackend() {
//        return backend(defaultDevice);
//    }
//
//    public Session activate() {
//        CURRENT.set(this);
//        return this;
//    }
//
//    public static Session current() {
//        Session session = CURRENT.get();
//        if (session == null) {
//            throw new IllegalStateException("No active session");
//        }
//        return session;
//    }
//
//    public static boolean hasCurrent() {
//        return CURRENT.get() != null;
//    }
//
//    @Override
//    public void close() {
//        if (CURRENT.get() == this) {
//            CURRENT.remove();
//        }
//        for (Backend b : backends.values()) {
//            b.close();
//        }
//    }
//
//    public static Builder builder() {
//        return new Builder();
//    }
//
//    public static final class Builder {
//        private final Map<Device, Backend> backends = new HashMap<>();
//        private Device defaultDevice;
//
//        private Builder() {}
//
//        public Builder register(Device device, Backend backend) {
//            backends.put(device, backend);
//            return this;
//        }
//
//        public Builder defaultDevice(Device device) {
//            this.defaultDevice = device;
//            return this;
//        }
//
//        public Session build() {
//            if (backends.isEmpty()) {
//                throw new IllegalStateException("At least one backend must be registered");
//            }
//            if (defaultDevice == null) {
//                defaultDevice = backends.keySet().iterator().next();
//            }
//            if (!backends.containsKey(defaultDevice)) {
//                throw new IllegalStateException("Default device has no registered backend: " + defaultDevice);
//            }
//            return new Session(backends, defaultDevice);
//        }
//    }
}
