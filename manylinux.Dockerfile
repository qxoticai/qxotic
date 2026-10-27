# The Linux build box for libtoknroll, libjinfer and their wheels: manylinux_2_28 (AlmaLinux 8, glibc 2.28)
# pins the glibc floor the wheels promise, the same on x86_64 and aarch64, whatever the host runs. GraalVM CE
# 25.0.4 is the floor the C shims need (make check-native-image), Maven the floor the poms enforce.
#   docker build -f manylinux.Dockerfile -t qxotic-manylinux .                              # x86_64
#   docker build -f manylinux.Dockerfile --build-arg ARCH=aarch64 -t qxotic-manylinux .     # on an arm64 host
#   docker run --rm -u "$(id -u):$(id -g)" -v /etc/passwd:/etc/passwd:ro -e HOME -v "$HOME/.m2:$HOME/.m2" -v "$HOME/.cache:$HOME/.cache" \
#       -v "$PWD:/src" -w /src qxotic-manylinux make -C jinfer/jinfer-native libjinfer check-wheel dist
# (passwd read-only: the JVM takes the home directory from the uid, and the model and Maven caches live under it)
ARG ARCH=x86_64
FROM quay.io/pypa/manylinux_2_28_${ARCH}
ARG ARCH
ARG GRAALVM_TAG=graal-25.4.4.1.1
ARG GRAALVM_FILE=graalvm-community-jdk-25i4-25.0.4.1.1
ARG MAVEN=3.9.16
# zip for the dist bundles; ffmpeg (RPM Fusion) for the image and audio codecs the jinfer checks exercise
RUN set -e; dnf install -y -q epel-release https://download1.rpmfusion.org/free/el/rpmfusion-free-release-8.noarch.rpm \
    && dnf install -y -q zip ffmpeg && dnf clean all; \
    case "$ARCH" in x86_64) g=x64 ;; *) g=$ARCH ;; esac; \
    url="https://github.com/graalvm/graalvm-ce-builds/releases/download/$GRAALVM_TAG/${GRAALVM_FILE}_linux-${g}_bin.tar.gz"; \
    curl -sSL -o /tmp/graalvm.tar.gz "$url" && curl -sSL -o /tmp/graalvm.sha256 "$url.sha256"; \
    echo "$(cut -d' ' -f1 /tmp/graalvm.sha256)  /tmp/graalvm.tar.gz" | sha256sum -c - >/dev/null; \
    mkdir /opt/graalvm && tar xzf /tmp/graalvm.tar.gz -C /opt/graalvm --strip-components=1; \
    curl -sSL "https://dlcdn.apache.org/maven/maven-3/$MAVEN/binaries/apache-maven-$MAVEN-bin.tar.gz" | tar xz -C /opt && mv /opt/apache-maven-$MAVEN /opt/maven; \
    rm /tmp/graalvm.tar.gz /tmp/graalvm.sha256
ENV JAVA_HOME=/opt/graalvm PATH=/opt/graalvm/bin:/opt/maven/bin:/opt/python/cp312-cp312/bin:$PATH
