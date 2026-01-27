package ai.qxotic.jota.backend;

import ai.qxotic.jota.Device;
import java.util.Set;

public interface BackendRegistry {
    void register(Backend backend);

    Backend backend(Device device);

    boolean hasBackend(Device device);

    Backend nativeBackend();

    Set<Device> devices();
}
