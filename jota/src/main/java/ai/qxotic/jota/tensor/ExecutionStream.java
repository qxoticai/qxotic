package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Device;
import java.util.Objects;

public record ExecutionStream(Device device, Object handle, boolean isDefault) {

    public ExecutionStream {
        Objects.requireNonNull(device, "device");
    }
}
