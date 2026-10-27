# The machines of the four wheel legs: `make wheels` ships a tarball of the ref to each over ssh, builds
# and checks both wheels there and brings them back. A leg with no host is skipped. Every host needs its
# toolchain on the PATH of a non-interactive ssh shell, or a SETUP line that puts it there.
HOST.linux-x86_64 = localhost
HOST.linux-aarch64 =
HOST.macos-arm64 = mac.lan
HOST.windows-x86_64 = win.lan

SETUP.macos-arm64 = export PATH=/opt/homebrew/bin:$$PATH JAVA_HOME=~/graalvm/graalvm-community-25.4.4.1.1+1.1/Contents/Home
