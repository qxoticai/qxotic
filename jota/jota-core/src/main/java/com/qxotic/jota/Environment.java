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

    // === Core properties (instance) ===

    /**
     * Host/native device used for JVM-side tensor access.
     *
     * <p>This device is always present. On JVM this is typically Panama-backed; on Native Image
     * this is typically C-backed.
     */
    Device nativeDevice();

    /**
     * Default execution device used when an operation does not explicitly choose a device.
     *
     * <p>This may differ from {@link #nativeDevice()}.
     */
    Device defaultDevice();

    /** Default floating-point type used by convenience tensor APIs. */
    DataType defaultFloat();

    /** Runtime registry for this environment. */
    RuntimeRegistry runtimes();

    // === Static factories ===

    /**
     * Creates an environment where native and default execution device are the same.
     *
     * <p>Use the 4-argument overload to keep host/native and default execution devices distinct.
     */
    static Environment of(Device defaultDevice, DataType defaultFloat, RuntimeRegistry runtimes) {
        return new EnvironmentImpl(defaultDevice, defaultDevice, defaultFloat, runtimes);
    }

    /**
     * Creates an environment with explicit native host device and default execution device.
     *
     * <p>{@code nativeDevice} should point to the guaranteed host-access runtime.
     */
    static Environment of(
            Device nativeDevice,
            Device defaultDevice,
            DataType defaultFloat,
            RuntimeRegistry runtimes) {
        return new EnvironmentImpl(nativeDevice, defaultDevice, defaultFloat, runtimes);
    }

    /** Creates an environment like the current one but with the given default device. */
    static Environment withDefaultDevice(Device device) {
        Environment c = current();
        return of(device, c.defaultFloat(), c.runtimes());
    }

    // === Static lifecycle ===

    /**
     * Returns the currently active environment.
     *
     * <p>If a scoped environment is active via {@link #with(Environment, Supplier)}, that scoped
     * environment is returned. Otherwise, the globally configured environment is returned.
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

    // === Static convenience methods (delegate to current()) ===

    /** Returns runtime for {@code device} from {@link #current()}. */
    static DeviceRuntime runtimeFor(Device device) {
        return current().runtimes().runtimeFor(device);
    }

    /** Returns the guaranteed host/native runtime from {@link #current()}. */
    static DeviceRuntime nativeRuntime() {
        return runtimeFor(current().nativeDevice());
    }

    /** Returns the default execution runtime from {@link #current()}. */
    static DeviceRuntime defaultRuntime() {
        return runtimeFor(current().defaultDevice());
    }

    /**
     * Returns the guaranteed host/native memory domain from {@link #current()}.
     *
     * <p>This domain is always {@link MemorySegment}-backed for host/JVM memory access.
     */
    @SuppressWarnings("unchecked")
    static MemoryDomain<MemorySegment> nativeMemoryDomain() {
        return (MemoryDomain<MemorySegment>) nativeRuntime().memoryDomain();
    }

    /**
     * Returns the memory domain for {@code device} from {@link #current()}.
     *
     * <p>Use {@link #nativeMemoryDomain()} when host/JVM access is required.
     */
    @SuppressWarnings("unchecked")
    static <B> MemoryDomain<B> memoryDomainFor(Device device) {
        Objects.requireNonNull(device, "device");
        return (MemoryDomain<B>) runtimeFor(device).memoryDomain();
    }

    /** Returns compute engine for {@code device} from {@link #current()}. */
    static ComputeEngine computeEngineFor(Device device) {
        return runtimeFor(device).computeEngine();
    }

    /** Returns runtime probe diagnostics from {@link #current()}. */
    static List<RuntimeDiagnostic> runtimeDiagnostics() {
        return current().runtimes().diagnostics();
    }

    static boolean hasRuntimeFor(Device device) {
        return current().runtimes().hasRuntimeFor(device);
    }
}
