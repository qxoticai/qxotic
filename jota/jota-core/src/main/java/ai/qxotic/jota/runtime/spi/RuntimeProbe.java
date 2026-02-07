package ai.qxotic.jota.runtime.spi;

import java.util.Objects;

public record RuntimeProbe(
        RuntimeProbeStatus status, String message, String hint, Throwable cause) {

    public RuntimeProbe {
        status = Objects.requireNonNull(status, "status");
        message = Objects.requireNonNullElse(message, "");
        hint = hint == null || hint.isBlank() ? null : hint;
    }

    public static RuntimeProbe available(String message) {
        return new RuntimeProbe(RuntimeProbeStatus.AVAILABLE, message, null, null);
    }

    public static RuntimeProbe missingSoftware(String message, String hint) {
        return new RuntimeProbe(RuntimeProbeStatus.MISSING_SOFTWARE, message, hint, null);
    }

    public static RuntimeProbe unsupportedHardware(String message, String hint) {
        return new RuntimeProbe(RuntimeProbeStatus.UNSUPPORTED_HARDWARE, message, hint, null);
    }

    public static RuntimeProbe misconfigured(String message, String hint, Throwable cause) {
        return new RuntimeProbe(RuntimeProbeStatus.MISCONFIGURED, message, hint, cause);
    }

    public static RuntimeProbe error(String message, Throwable cause) {
        return new RuntimeProbe(RuntimeProbeStatus.ERROR, message, null, cause);
    }

    public boolean isAvailable() {
        return status == RuntimeProbeStatus.AVAILABLE;
    }
}
