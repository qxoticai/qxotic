package ai.qxotic.format.safetensors;

/**
 * Unchecked exception thrown when safetensors format is invalid or corrupted.
 *
 * <p>This indicates the file structure violates the safetensors specification, such as:
 *
 * <ul>
 *   <li>Invalid header structure
 *   <li>Missing required fields (dtype, shape, data_offsets)
 *   <li>Type mismatches in JSON metadata
 *   <li>Invalid tensor offsets or sizes
 *   <li>Overlapping tensor data
 * </ul>
 */
public class SafetensorsFormatException extends RuntimeException {

    /**
     * Creates a format exception with message.
     *
     * @param message error description
     */
    public SafetensorsFormatException(String message) {
        super(message);
    }

    /**
     * Creates a format exception with message and cause.
     *
     * @param message error description
     * @param cause root cause
     */
    public SafetensorsFormatException(String message, Throwable cause) {
        super(message, cause);
    }
}
