/*
 * A compile-only declaration matching the NAME of GraalVM's svm-shared @AlwaysInline: native-image
 * binds the annotation by its fully qualified name at image build time, and HotSpot never sees
 * this type. Written here; no GraalVM source is included, and this module is never published.
 */
package com.oracle.svm.shared;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/** Forces the native-image compiler to inline the annotated method; the value documents why. */
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.METHOD, ElementType.CONSTRUCTOR})
public @interface AlwaysInline {
    String value();
}
