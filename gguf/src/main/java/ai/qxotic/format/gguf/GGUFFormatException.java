package ai.qxotic.format.gguf;

/**
 * Unchecked exception thrown when GGUF format is invalid or corrupted.
 *
 * <p>This indicates the file structure violates the GGUF specification, such as:
 *
 * <ul>
 *   <li>Invalid magic number
 *   <li>Unsupported version
 *   <li>Invalid metadata types
 *   <li>Nested arrays (not supported by spec)
 * </ul>
 */
public class GGUFFormatException extends RuntimeException {

    public GGUFFormatException(String message) {
        super(message);
    }

    public GGUFFormatException(String message, Throwable cause) {
        super(message, cause);
    }
}
