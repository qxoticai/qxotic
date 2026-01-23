package ai.qxotic.jota;

import java.util.Objects;

public interface Device {
    String name();

    String leafName();

    Device parent();

    Device root();

    static Device of(String rootName) {
        return new DeviceImpl(null, rootName);
    }

    default Device child(String childName) {
        return new DeviceImpl(this, childName);
    }

    // Roots
    Device CPU = Device.of("cpu");
    Device GPU = Device.of("gpu");

    // Engines
    Device PANAMA = CPU.child("panama"); // MemorySegment-backed memory
    Device CUDA = GPU.child("cuda");

    static Device defaultDevice() {
        return Environment.current().defaultDevice();
    }

    // CUDA.belongsTo(Device.GPU)
    default boolean belongsTo(Device other) {
        return this.name().startsWith(other.name());
    }
}

final class DeviceImpl implements Device {
    final Device parent;
    final String leafName;
    final String fullName;

    private static final String SEPARATOR = ":";

    DeviceImpl(Device parent, String leafName) {
        this.parent = parent;
        if (leafName == null || leafName.trim().isEmpty()) {
            throw new IllegalArgumentException("child name must not be null or empty");
        }
        if (leafName.contains(SEPARATOR)) {
            throw new IllegalArgumentException("child name cannot contain separator " + SEPARATOR);
        }
        this.leafName = leafName.trim().toLowerCase();
        this.fullName = (parent == null) ? this.leafName : parent.name() + ":" + this.leafName;
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
    public String leafName() {
        return leafName;
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
