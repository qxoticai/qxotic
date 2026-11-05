package com.qxotic.toknroll.libtoknroll;

import java.util.function.BooleanSupplier;

/**
 * Whether this image is libtoknroll's, built lean: the substitutions it gates trade a JDK facility
 * the library barely uses for the code it drags in. libjinfer, built from these classes too, keeps
 * the JDK's (it verifies gigabytes with the intrinsic SHA-256). The pom says, with
 * -Dtoknroll.image.lean=true.
 */
final class Lean implements BooleanSupplier {
    @Override
    public boolean getAsBoolean() {
        return Boolean.getBoolean("toknroll.image.lean");
    }
}
