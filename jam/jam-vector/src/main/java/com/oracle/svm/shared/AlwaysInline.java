/*
 * Vendored from GraalVM's svm-shared module (GPLv2 with Classpath Exception), verbatim shape:
 * @Retention(RUNTIME) @Target({METHOD, CONSTRUCTOR}) with a single String value().
 *
 * WHY VENDORED: native-image's hosted inlining matches this annotation BY NAME from the app
 * classpath bytecode; the artifact itself is not published to Maven Central (only GraalVM dev
 * builds ship it), so depending on it broke the build for anyone without a locally installed
 * snapshot. Shipping the one annotation class keeps the build self-contained while remaining
 * recognized by the GraalVM 25.1+ builders that look for com.oracle.svm.shared.AlwaysInline.
 * It is inert on HotSpot and on builders that predate the name.
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
