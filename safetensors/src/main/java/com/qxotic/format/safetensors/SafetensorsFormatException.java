package com.qxotic.format.safetensors;

/** Unchecked exception for invalid or corrupted safetensors format. */
public class SafetensorsFormatException extends RuntimeException {

    public SafetensorsFormatException(String message) {
        super(message);
    }

    public SafetensorsFormatException(String message, Throwable cause) {
        super(message, cause);
    }
}
