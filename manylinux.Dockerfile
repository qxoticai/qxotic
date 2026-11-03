# The Linux build box for libtoknroll, libjinfer and their wheels: manylinux_2_28 (AlmaLinux 8, glibc 2.28)
# pins the glibc floor the wheels promise, the same on x86_64 and aarch64, whatever the host runs. Oracle GraalVM,
# 25.0.4 or later being the floor the C shims need (make check-native-image), records the profile libtoknroll ships
# built with (PGO); Maven is the floor the poms enforce.
#   make wheels          # on Linux: builds this image for the host's architecture, then both wheels and their checks inside it
#   make wheels-linux    # the same on any host with docker: a Mac's builds Linux aarch64
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
ARG MAVEN=3.9.16
# zip for the dist bundles; ffmpeg (RPM Fusion) for the image and audio codecs the jinfer checks exercise
RUN set -e; dnf install -y -q epel-release https://download1.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm \
    && dnf install -y -q zip ffmpeg && dnf clean all; \
    case "$ARCH" in x86_64) g=x64 ;; *) g=$ARCH ;; esac; \
    case "$ARCH" in x86_64) sha=$GRAALVM_SHA256_x86_64 ;; *) sha=$GRAALVM_SHA256_aarch64 ;; esac; \
    curl -sSL -o /tmp/graalvm.tar.gz "https://gds.oracle.com/download/graal/${GRAALVM}_linux-${g}_bin.tar.gz"; \
    echo "$sha  /tmp/graalvm.tar.gz" | sha256sum -c - >/dev/null; \
    mkdir /opt/graalvm && tar xzf /tmp/graalvm.tar.gz -C /opt/graalvm --strip-components=1; \
    curl -sSL "https://dlcdn.apache.org/maven/maven-3/$MAVEN/binaries/apache-maven-$MAVEN-bin.tar.gz" | tar xz -C /opt && mv /opt/apache-maven-$MAVEN /opt/maven; \
    rm /tmp/graalvm.tar.gz
ENV JAVA_HOME=/opt/graalvm PATH=/opt/graalvm/bin:/opt/maven/bin:/opt/python/cp312-cp312/bin:$PATH
