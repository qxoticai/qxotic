# The machines of the four wheel legs: `make wheels` ships a tarball of the ref to each over ssh, builds
# and checks both wheels there and brings them back. A leg with no host is skipped. Every host needs its
# toolchain on the PATH of a non-interactive ssh shell, or a SETUP line that puts it there.
HOST.linux-x86_64 = localhost
HOST.linux-aarch64 = mac.lan
HOST.macos-arm64 = mac.lan
HOST.windows-x86_64 = win.lan

SETUP.macos-arm64 = export PATH=/opt/homebrew/bin:$$PATH JAVA_HOME=~/graalvm/graalvm-jdk-25.0.4+7.1/Contents/Home
# Linux aarch64 builds in the manylinux container on the Mac, docker served by Colima (brew install colima
# docker docker-buildx; colima start --vm-type vz --mount-type virtiofs --cpu 8 --memory 20 --mount "$HOME:w")
SETUP.linux-aarch64 = export PATH=/opt/homebrew/bin:$$PATH
