package com.qxotic.jinfer.codecs;

import java.io.IOException;
import java.time.Duration;
import java.util.List;
import java.util.Locale;

/**
 * The media backend-selection machinery shared by {@link AudioCodec} and {@link ImageCodec}: the
 * native-image probe, the {@code -D} property override, and reflective instantiation of the
 * java.desktop-backed decoders.
 */
final class Codecs {

    private Codecs() {}

    /**
     * Inside a native image? Read at RUN time: a static-final answer keyed off this property gets
     * constant-folded at build time (the {@code Arenas} law), which is exactly wrong - a native
     * binary must detect its host.
     */
    static boolean nativeImage() {
        return System.getProperty("org.graalvm.nativeimage.imagecode") != null;
    }

    /**
     * The configured backend name, normalized: the {@code -D} property value lowercased, else
     * ffmpeg - always in a native image, which has no {@code java.desktop} decoders, and on a JVM
     * with ffmpeg on PATH, so both runtimes read every format alike - else the JDK's own {@code
     * jdkBackend}. A selected backend is the only one asked.
     */
    static String choice(String property, String jdkBackend) {
        String c = System.getProperty(property);
        if (c != null && !c.isBlank()) return c.toLowerCase(Locale.ROOT);
        return nativeImage() || ffmpegOnPath() ? "ffmpeg" : jdkBackend;
    }

    /** Whether {@code ffmpeg -version} runs: asked once per facade, when it picks its decoder. */
    static boolean ffmpegOnPath() {
        try {
            Subprocess.run(List.of("ffmpeg", "-version"), null, Duration.ofSeconds(2), 64 << 10);
            return true;
        } catch (IOException missingOrMute) {
            return false;
        }
    }

    /**
     * Instantiate a backend by NON-CONSTANT class name: a literal at the {@code Class.forName} site
     * would let native-image fold the call and drag the module ({@code java.desktop}, {@code
     * javax.sound.sampled}) into the image. A backend this runtime lacks is refused, naming the
     * {@code property} switch to ffmpeg: the choice is never swapped behind the caller's back.
     */
    static <T> T reflect(String className, String property) {
        try {
            @SuppressWarnings("unchecked")
            T backend = (T) Class.forName(className).getDeclaredConstructor().newInstance();
            return backend;
        } catch (ReflectiveOperationException | LinkageError e) {
            throw new IllegalStateException(
                    className
                            + " is unavailable in this runtime ("
                            + e
                            + "): -D"
                            + property
                            + "=ffmpeg decodes without it",
                    e);
        }
    }
}
