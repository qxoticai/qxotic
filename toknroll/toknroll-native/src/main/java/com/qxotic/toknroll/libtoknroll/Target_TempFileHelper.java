package com.qxotic.toknroll.libtoknroll;

import com.oracle.svm.core.annotate.Substitute;
import com.oracle.svm.core.annotate.TargetClass;
import java.io.IOException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileAttribute;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Files.createTempFile's helper, without the JDK's SecureRandom behind the names: a SecureRandom is
 * the whole security provider list, half a megabyte, for a name in the caller's own cache folder
 * that only has to be new. As the JDK's: created exclusively, retried while the name is taken,
 * private to the user on POSIX when the caller gave no attributes.
 */
@TargetClass(className = "java.nio.file.TempFileHelper", onlyWith = Lean.class)
@Substitute
final class Target_java_nio_file_TempFileHelper {
    private static final AtomicLong NEXT = new AtomicLong(System.nanoTime());

    @Substitute
    static Path createTempFile(Path dir, String prefix, String suffix, FileAttribute<?>[] attrs)
            throws IOException {
        return create(dir, prefix, suffix == null ? ".tmp" : suffix, false, attrs);
    }

    @Substitute
    static Path createTempDirectory(Path dir, String prefix, FileAttribute<?>[] attrs)
            throws IOException {
        return create(dir, prefix, "", true, attrs);
    }

    private static Path create(
            Path dir, String prefix, String suffix, boolean directory, FileAttribute<?>[] attrs)
            throws IOException {
        if (dir == null) dir = Path.of(System.getProperty("java.io.tmpdir"));
        if (prefix == null) prefix = "";
        if (attrs.length == 0 && Files.getFileStore(dir).supportsFileAttributeView("posix"))
            attrs =
                    new FileAttribute<?>[] {
                        PosixFilePermissions.asFileAttribute(
                                PosixFilePermissions.fromString(
                                        directory ? "rwx------" : "rw-------"))
                    };
        for (; ; ) {
            Path name =
                    dir.getFileSystem()
                            .getPath(
                                    prefix
                                            + Long.toUnsignedString(NEXT.incrementAndGet())
                                            + suffix);
            if (name.getParent() != null)
                throw new IllegalArgumentException("Invalid prefix or suffix");
            Path path = dir.resolve(name);
            try {
                return directory
                        ? Files.createDirectory(path, attrs)
                        : Files.createFile(path, attrs);
            } catch (FileAlreadyExistsException taken) {
                // the next number
            }
        }
    }
}
