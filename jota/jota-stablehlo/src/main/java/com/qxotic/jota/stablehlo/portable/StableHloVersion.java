package com.qxotic.jota.stablehlo.portable;

/**
 * A StableHLO / VHLO version {@code major.minor.patch} (see {@code stablehlo/dialect/Version.h}).
 *
 * <p>Portable artifacts are guaranteed readable when their version lies in [{@link #MINIMUM},
 * {@link #CURRENT}]; {@code CURRENT} tracks the StableHLO revision this module was written against
 * (recorded in {@code src/test/resources/stablehlo-samples/SAMPLES_VERSION}).
 */
public record StableHloVersion(int major, int minor, int patch)
        implements Comparable<StableHloVersion> {

    /** Oldest VHLO version StableHLO can still deserialize. */
    public static final StableHloVersion MINIMUM = new StableHloVersion(0, 9, 0);

    /** VHLO version of the StableHLO revision this model follows. */
    public static final StableHloVersion CURRENT = new StableHloVersion(1, 27, 1);

    public StableHloVersion {
        if (major < 0 || minor < 0 || patch < 0) {
            throw new IllegalArgumentException("negative version component");
        }
    }

    /** Parses {@code "1.2.3"}. */
    public static StableHloVersion parse(String text) {
        String[] parts = text.trim().split("\\.");
        if (parts.length != 3) {
            throw new IllegalArgumentException("expected major.minor.patch, got " + text);
        }
        return new StableHloVersion(
                Integer.parseInt(parts[0]), Integer.parseInt(parts[1]), Integer.parseInt(parts[2]));
    }

    /** True when an artifact of this version is within the supported compatibility window. */
    public boolean isSupported() {
        return compareTo(MINIMUM) >= 0 && compareTo(CURRENT) <= 0;
    }

    @Override
    public int compareTo(StableHloVersion o) {
        if (major != o.major) {
            return Integer.compare(major, o.major);
        }
        if (minor != o.minor) {
            return Integer.compare(minor, o.minor);
        }
        return Integer.compare(patch, o.patch);
    }

    @Override
    public String toString() {
        return major + "." + minor + "." + patch;
    }
}
