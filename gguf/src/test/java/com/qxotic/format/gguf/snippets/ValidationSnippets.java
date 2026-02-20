package com.qxotic.format.gguf.snippets;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

/**
 * Snippets for validation.
 */
@SuppressWarnings("unused")
public class ValidationSnippets {

    // --8<-- [start:quick-validate]
    public static boolean isValidGGUF(Path path) {
        try {
            GGUF gguf = GGUF.read(path);
            return gguf.getTensors().size() > 0;
        } catch (Exception e) {
            return false;
        }
    }
    // --8<-- [end:quick-validate]

    // --8<-- [start:comprehensive-validate]
    public static ValidationResult validate(Path path) {
        List<String> errors = new ArrayList<>();
        List<String> warnings = new ArrayList<>();

        try {
            GGUF gguf = GGUF.read(path);

            if (!gguf.containsKey("general.architecture")) {
                warnings.add("Missing architecture");
            }

            for (TensorEntry tensor : gguf.getTensors()) {
                if (tensor.name() == null || tensor.name().isEmpty()) {
                    errors.add("Empty tensor name");
                }
                if (tensor.byteSize() <= 0) {
                    errors.add("Invalid size: " + tensor.name());
                }
                if (tensor.absoluteOffset(gguf) % gguf.getAlignment() != 0) {
                    warnings.add("Unaligned: " + tensor.name());
                }
            }

        } catch (Exception e) {
            errors.add("Parse error: " + e.getMessage());
        }

        return new ValidationResult(errors.isEmpty(), errors, warnings);
    }

    public static class ValidationResult {
        public final boolean valid;
        public final List<String> errors;
        public final List<String> warnings;

        public ValidationResult(boolean valid, List<String> errors, List<String> warnings) {
            this.valid = valid;
            this.errors = errors;
            this.warnings = warnings;
        }
    }
    // --8<-- [end:comprehensive-validate]
}
