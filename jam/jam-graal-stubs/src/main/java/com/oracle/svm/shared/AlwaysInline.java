/*
 * Compile-only stub of GraalVM's svm-shared @AlwaysInline (GPLv2 with Classpath Exception), verbatim
 * shape: @Retention(RUNTIME) @Target({METHOD, CONSTRUCTOR}) with a single String value().
 *
 * WHY A STUB MODULE: native-image's hosted inlining matches this annotation BY NAME from the app
 * classpath bytecode. GraalVM's real svm-shared artifact is not published to Maven Central (only dev
 * builds ship it), so depending on it broke the build for anyone without a local snapshot. This
 * module ships the one annotation class so downstream code compiles anywhere; consumers depend on it
 * with `provided` scope, so it is NOT bundled into their jars. On a GraalVM 25.1+ native-image build
 * the real com.oracle.svm.shared.AlwaysInline is supplied by the image builder and honored; on
 * HotSpot the annotation is inert (its class is simply absent from the runtime classpath, and nothing
 * reflects on it).
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
