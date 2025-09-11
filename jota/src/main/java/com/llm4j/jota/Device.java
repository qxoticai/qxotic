package com.llm4j.jota;

import java.util.Objects;

/**
 * Represents a hardware device where memory can be allocated and computations can be executed.
 * Used to distinguish between different execution contexts (e.g., CPU, GPU, TPU) for
 * heterogeneous computing.
 */
public interface Device {

    /**
     * Returns the name of the device in a lowercase, platform-neutral format (e.g., "cpu", "gpu").
     * This name is typically used for debugging, logging, or comparing devices.
     *
     * @return The canonical name of the device (never {@code null} or empty).
     * @example {@code device.name().equals("cpu")}
     */
    String name();

    /**
     * Returns the root ancestor of this device (the topmost parent in the hierarchy).
     * For root devices, returns the device itself.
     *
     * @return The root device in this device's hierarchy (never {@code null}).
     * @example {@code device.root()} returns the CPU device for any device under "cpu:*" hierarchy
     */
    String localName();

    /**
     * Returns the root ancestor of this device (the topmost parent in the hierarchy).
     * For root devices, returns the device itself.
     *
     * @return The root device in this device's hierarchy (never {@code null}).
     * @example {@code device.root()} returns the CPU device for any device under "cpu:*" hierarchy
     */
    Device parent();

    /**
     * Returns the root ancestor of this device (the topmost parent in the hierarchy).
     * For root devices, returns the device itself.
     *
     * @return The root device in this device's hierarchy (never {@code null}).
     * @example {@code device.root()} returns the CPU device for any device under "cpu:*" hierarchy
     */
    Device root();

    /**
     * Creates a root device with the given name (converted to lowercase).
     *
     * @param rootName Root device name (e.g., "cpu", "gpu").
     * @return Root device instance.
     * @throws IllegalArgumentException if name is null or empty.
     */
    static Device of(String rootName) {
        return new DeviceImpl(null, rootName);
    }

    /**
     * Creates a child device under the specified parent.
     *
     * @param childName Name for the new child (converted to lowercase).
     * @return Child device instance.
     * @throws IllegalArgumentException if childName is null or empty.
     * @throws NullPointerException     if parent is null.
     */
    static Device child(Device parent, String childName) {
        Objects.requireNonNull(parent);
        return new DeviceImpl(parent, childName);
    }

    /**
     * A built-in {@link Device} instance representing a CPU. This is the default device
     * for most operations unless explicitly overridden.
     */
    Device CPU = Device.of("cpu");

    /**
     * A built-in {@link Device} instance representing a GPU.
     */
    Device GPU = Device.of("gpu");
}

final class DeviceImpl implements Device {
    final Device parent;
    final String localName;
    final String fullName;

    private static final String SEPARATOR = ":";

    DeviceImpl(Device parent, String localName) {
        this.parent = parent;
        if (localName == null || localName.trim().isEmpty()) {
            throw new IllegalArgumentException("child name must not be null or empty");
        }
        if (localName.contains(SEPARATOR)) {
            throw new IllegalArgumentException("child name cannot contain separator " + SEPARATOR);
        }
        this.localName = localName.trim().toLowerCase();
        this.fullName = (parent == null) ? this.localName : parent.name() + ":" + this.localName;
    }

    @Override
    public Device parent() {
        return this.parent;
    }

    @Override
    public Device root() {
        Device root = this;
        while (root.parent() != null) {
            root = root.parent();
        }
        return root;
    }

    @Override
    public String localName() {
        return localName;
    }

    @Override
    public String name() {
        return fullName;
    }

    @Override
    public boolean equals(Object other) {
        return (other instanceof DeviceImpl that && this.name().equals(that.name()));
    }

    @Override
    public int hashCode() {
        return Objects.hashCode(fullName);
    }

    @Override
    public String toString() {
        return name();
    }
}