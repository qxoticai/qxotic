package com.qxotic.jota;

import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.runtime.ComputeEngine;
import com.qxotic.jota.runtime.DeviceRuntime;
import com.qxotic.jota.runtime.RuntimeDiagnostic;
import com.qxotic.jota.runtime.RuntimeRegistry;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Objects;
import java.util.function.Supplier;

public interface Environment {

    // === Core properties ===

    Device nativeDevice();

    Device defaultDevice();

    DataType defaultFloat();

    RuntimeRegistry runtimes();

    // === Derived instance methods ===

    default DeviceRuntime runtimeFor(Device device) {
        return runtimes().runtimeFor(device);
    }

    default DeviceRuntime nativeRuntime() {
        return runtimeFor(nativeDevice());
    }

    default DeviceRuntime defaultRuntime() {
        return runtimeFor(defaultDevice());
    }

    @SuppressWarnings("unchecked")
    default MemoryDomain<MemorySegment> nativeMemoryDomain() {
        return (MemoryDomain<MemorySegment>) nativeRuntime().memoryDomain();
    }

    @SuppressWarnings("unchecked")
    default <B> MemoryDomain<B> memoryDomainFor(Device device) {
        Objects.requireNonNull(device, "device");
        return (MemoryDomain<B>) runtimeFor(device).memoryDomain();
    }

    default ComputeEngine computeEngineFor(Device device) {
        return runtimeFor(device).computeEngine();
    }

    default List<RuntimeDiagnostic> runtimeDiagnostics() {
        return runtimes().diagnostics();
    }

    // === Static factories ===

    static Environment of(Device defaultDevice, DataType defaultFloat, RuntimeRegistry runtimes) {
        return new EnvironmentImpl(defaultDevice, defaultDevice, defaultFloat, runtimes);
    }

    static Environment of(
            Device nativeDevice,
            Device defaultDevice,
            DataType defaultFloat,
            RuntimeRegistry runtimes) {
        return new EnvironmentImpl(nativeDevice, defaultDevice, defaultFloat, runtimes);
    }

    // === Static lifecycle ===

    /**
     * Returns the currently active environment.
     *
     * <p>If a scoped environment is active via {@link #with(Environment, Supplier)}, that scoped
     * environment is returned. Otherwise the globally configured environment is returned.
     */
    static Environment current() {
        return EnvironmentImpl.current();
    }

    /** Returns the globally configured environment, or the default built-in global environment. */
    static Environment global() {
        return EnvironmentImpl.global();
    }

    /**
     * Configures the process-wide global environment.
     *
     * <p>This method can only be called once. Subsequent calls fail.
     */
    static void configureGlobal(Environment environment) {
        EnvironmentImpl.configureGlobal(environment);
    }

    /**
     * Executes {@code action} with {@code environment} bound as the current scoped environment.
     *
     * <p>The binding applies only within this call and does not mutate global configuration.
     */
    static <T> T with(Environment environment, Supplier<T> action) {
        return EnvironmentImpl.with(environment, action);
    }
}
