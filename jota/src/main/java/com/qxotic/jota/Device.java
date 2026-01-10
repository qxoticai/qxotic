package com.qxotic.jota;

import java.util.Objects;

public interface Device {
    String name();

    String localName();

    Device parent();

    Device root();

    static Device of(String rootName) {
        return new DeviceImpl(null, rootName);
    }

    static Device child(Device parent, String childName) {
        Objects.requireNonNull(parent);
        return new DeviceImpl(parent, childName);
    }

    default Device child(String childName) {
        return new DeviceImpl(this, childName);
    }

    Device CPU = Device.of("cpu");

    Device JAVA = CPU.child("java"); // Java managed heap
    Device NATIVE = CPU.child("native"); // native memory address space

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