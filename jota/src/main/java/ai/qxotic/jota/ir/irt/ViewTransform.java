package ai.qxotic.jota.ir.irt;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import java.util.Set;

/**
 * View transform operation in IR-T. Represents operations that only change the layout (shape +
 * stride) without allocating new memory. The hint indicates the type of transform: "broadcast",
 * "transpose", "slice", "view", "expand".
 */
public record ViewTransform(IRTNode input, String hint, Layout layout) implements IRTNode {

    private static final Set<String> VALID_HINTS =
            Set.of("broadcast", "transpose", "slice", "view", "expand");

    public ViewTransform {
        if (input == null) {
            throw new IllegalArgumentException("input cannot be null");
        }
        if (hint == null || hint.isEmpty()) {
            throw new IllegalArgumentException("hint cannot be null or empty");
        }
        if (!VALID_HINTS.contains(hint)) {
            throw new IllegalArgumentException(
                    "Invalid hint: " + hint + ", must be one of: " + VALID_HINTS);
        }
        if (layout == null) {
            throw new IllegalArgumentException("layout cannot be null");
        }
    }

    @Override
    public DataType dataType() {
        return input.dataType();
    }
}
