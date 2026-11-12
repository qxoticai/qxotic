/*
 * A compile-only declaration matching the NAME of GraalVM's svm-shared @Uninterruptible: native-image
 * binds the annotation by its fully qualified name at image build time, and HotSpot never sees this
 * type, so it declares only the element this repository sets; the builder's own has more, with defaults.
 * Written here; no GraalVM source is included, and this module is never published.
 */
package com.oracle.svm.shared;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Code the native-image runtime runs where no safepoint may come, an entry point's exception
 * handler among it: it must not allocate, and calls only code that is uninterruptible too.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.METHOD, ElementType.CONSTRUCTOR})
public @interface Uninterruptible {
    String reason();
}
