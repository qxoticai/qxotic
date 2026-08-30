/*
 * Compile-only stub of GraalVM's svm-core @NeverInline (GPLv2 with Classpath Exception), verbatim
 * shape: @Retention(RUNTIME) @Target({METHOD, CONSTRUCTOR}) with a single String value(). Same
 * reasoning as the AlwaysInline stub beside it: native-image honors the real class by name, HotSpot
 * never sees it.
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
