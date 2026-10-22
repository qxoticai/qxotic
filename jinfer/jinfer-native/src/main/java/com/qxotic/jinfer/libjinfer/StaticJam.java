package com.qxotic.jinfer.libjinfer;

import com.oracle.svm.hosted.c.NativeLibraries;
import org.graalvm.nativeimage.hosted.Feature;

/**
 * Links libjam into the image as a built-in JNI library, the way the JDK's own native libraries
 * are: {@code System.loadLibrary("jam")} is then answered from the image and no library is
 * extracted at run time. The archive is jam-native's {@code jam_static} target, found through
 * {@code -H:CLibraryPath}; the bundled per-platform resources are excluded so NativeLoader falls
 * through to {@code loadLibrary}. The image treats a library as built in only when it defines
 * {@code JNI_OnLoad_<name>}, which jam_jni.c provides; without it loadLibrary looks for a libjam.so
 * on disk and the native provider silently disappears.
 */
public final class StaticJam implements Feature {
    @Override
    public void beforeAnalysis(BeforeAnalysisAccess access) {
        NativeLibraries.singleton().addStaticJniLibrary("jam");
    }
}
