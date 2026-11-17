# The Linux build box for libtoknroll, libjinfer, their wheels and release archives, the jinfer command, and the
# shipped linux-aarch64 libjam: manylinux_2_28 (AlmaLinux 8, glibc 2.28) pins the glibc floor the wheels and archives
# promise, the same on x86_64 and aarch64, whatever the host runs. Oracle GraalVM, 25.0.4 or later being the floor
# the C shims need (make check-native-image), records the profile libtoknroll ships built with (PGO); Maven is the
# floor the poms enforce; zig compiles libjam against the same 2.28 (jam/jam-native/scripts/natives.sh).
#   make wheels          # on Linux: builds this image for the host's architecture, then both wheels and their checks inside it
#   make dists           # on Linux: the same for the three release archives
#   make wheels-linux    # the same on any host with docker: a Mac's builds Linux aarch64 (make dists-linux, the archives)
#   make jam-natives     # on a Mac: libjam for darwin-aarch64 natively, then for linux-aarch64 in here
# (the container runs as the caller with /etc/passwd read-only: the JVM takes the home directory from the uid,
# and the model and Maven caches live under it)
ARG ARCH=x86_64
FROM quay.io/pypa/manylinux_2_28_${ARCH}
ARG ARCH
# Oracle GraalVM 25.4 (JDK 25.0.4.1.1), from Oracle's GDS, which publishes no checksum: the archives' SHA-256,
# x86_64 and aarch64, are pinned here (the same as Arch's jdk25-graalvm-bin)
ARG GRAALVM=25i4/archive/graalvm-jdk-25i4-25.0.4.1.1
ARG GRAALVM_SHA256_x86_64=4fcc632cfc68e98f49f9316f8a3588bafe4f5289f120104e2d290a75cf33e28e
ARG GRAALVM_SHA256_aarch64=780d5784d3db9bbcff77775c4d026e4a0e21a041ba9dba7c59135949ea5234a4
# Maven from the Apache archive, which keeps every release (the mirrors drop all but the current ones), its
# SHA-512 as Apache publishes it
ARG MAVEN=3.9.16
ARG MAVEN_SHA512=831a8591fe20c8243b1dbe7d71e3244f31d1665b0804b2e825e38cbbe5ce0cafb8338851f90780735568773e0a6cd07bbec107cda0b896b008b861075358b6f6
# zig 0.16.0, the version natives.sh expects; ziglang.org publishes these checksums
ARG ZIG=0.16.0
ARG ZIG_SHA256_x86_64=70e49664a74374b48b51e6f3fdfbf437f6395d42509050588bd49abe52ba3d00
ARG ZIG_SHA256_aarch64=ea4b09bfb22ec6f6c6ceac57ab63efb6b46e17ab08d21f69f3a48b38e1534f17
# ffmpeg (RPM Fusion) for the image and audio codecs the jinfer checks exercise, espeak-ng for the speech ones
RUN set -eu -o pipefail; \
    dnf install -y -q epel-release https://download1.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm; \
    dnf install -y -q ffmpeg espeak-ng; \
    dnf clean all; \
    case "$ARCH" in x86_64) g=x64 ;; *) g=$ARCH ;; esac; \
    case "$ARCH" in x86_64) sha=$GRAALVM_SHA256_x86_64; zsha=$ZIG_SHA256_x86_64 ;; *) sha=$GRAALVM_SHA256_aarch64; zsha=$ZIG_SHA256_aarch64 ;; esac; \
    curl -fsSL -o /tmp/graalvm.tar.gz "https://gds.oracle.com/download/graal/${GRAALVM}_linux-${g}_bin.tar.gz"; \
    echo "$sha  /tmp/graalvm.tar.gz" | sha256sum -c - >/dev/null; \
    mkdir /opt/graalvm; \
    tar xzf /tmp/graalvm.tar.gz -C /opt/graalvm --strip-components=1; \
    curl -fsSL -o /tmp/maven.tar.gz "https://archive.apache.org/dist/maven/maven-3/$MAVEN/binaries/apache-maven-$MAVEN-bin.tar.gz"; \
    echo "$MAVEN_SHA512  /tmp/maven.tar.gz" | sha512sum -c - >/dev/null; \
    mkdir /opt/maven; \
    tar xzf /tmp/maven.tar.gz -C /opt/maven --strip-components=1; \
    curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/$ZIG/zig-$ARCH-linux-$ZIG.tar.xz"; \
    echo "$zsha  /tmp/zig.tar.xz" | sha256sum -c - >/dev/null; \
    mkdir /opt/zig; \
    tar xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1; \
    rm /tmp/graalvm.tar.gz /tmp/maven.tar.gz /tmp/zig.tar.xz
ENV JAVA_HOME=/opt/graalvm PATH=/opt/graalvm/bin:/opt/maven/bin:/opt/zig:/opt/python/cp312-cp312/bin:$PATH
# The checkout is mounted at /src. The container runs as the caller, but a Mac's VM shows the mount under
# another owner, and git refuses a repository it thinks is someone else's; natives.sh needs git there to
# stamp the library with its commit.
RUN git config --system --add safe.directory /src
