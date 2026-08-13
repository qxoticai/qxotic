package com.qxotic.jinfer.x.boundary.media;

import java.util.Locale;
import java.util.function.Supplier;

/**
 * The media backend-selection machinery shared by {@link AudioCodec} and {@link ImageCodec}: the
 * native-image probe, the {@code -D} property override, and reflective instantiation of the
 * java.desktop-backed decoders.
 */
final class Codecs {

    static final System.Logger LOG = System.getLogger("jinfer.media");

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
     * The configured backend name, normalized: the {@code -D} property value lowercased, or the
     * platform default - ffmpeg under native-image (no java.desktop SPI discovery there), {@code
     * jvmDefault} on a JVM.
     */
    static String choice(String property, String jvmDefault) {
        String c = System.getProperty(property);
        return c == null || c.isBlank()
                ? (nativeImage() ? "ffmpeg" : jvmDefault)
                : c.toLowerCase(Locale.ROOT);
    }

    /**
     * Instantiate a backend by NON-CONSTANT class name: a literal at the {@code Class.forName} site
     * would let native-image fold the call and drag the module ({@code java.desktop}, {@code
     * javax.sound.sampled}) into the image. Falls back to ffmpeg when the backend cannot load.
     */
    static <T> T reflect(String className, Supplier<T> ffmpegFallback) {
        try {
            @SuppressWarnings("unchecked")
            T backend = (T) Class.forName(className).getDeclaredConstructor().newInstance();
            return backend;
        } catch (ReflectiveOperationException | LinkageError e) {
            LOG.log(
                    System.Logger.Level.WARNING,
                    "media backend ''{0}'' unavailable ({1}); falling back to ffmpeg",
                    className,
                    e);
            return ffmpegFallback.get();
        }
    }
}
