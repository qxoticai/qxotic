/*
 * A compile-only declaration matching the NAME of GraalVM's svm-core @NeverInline: native-image
 * binds the annotation by its fully qualified name at image build time, and HotSpot never sees
 * this type. Written here; no GraalVM source is included, and this module is never published.
 */
package com.oracle.svm.core;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * Forbids the native-image compiler from inlining the annotated method: a Vector API kernel that
 * takes and returns memory keeps its own expansion budget instead of joining its caller's.
 */
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.METHOD, ElementType.CONSTRUCTOR})
public @interface NeverInline {
    String value();
}
