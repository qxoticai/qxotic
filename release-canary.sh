#!/usr/bin/env bash
# Release canary: prove the artifacts a consumer would actually GET are usable.
#
#   1. install the release build (flattened poms, sources, javadocs) of the BOM plus the
#      langchain4j integration closure into a THROWAWAY local repository
#   2. compile a minimal consumer against ONLY that repository: it imports jinfer-bom,
#      declares jinfer-langchain4j WITHOUT a version, and makes one representative
#      builder call
#
# A green reactor can still ship a broken BOM or flattened pom; this catches broken
# dependencyManagement, leaked test-scope deps and flatten-plugin regressions.
# Not part of the default build: the empty repository re-downloads plugins and
# third-party deps (slow, online). Run it before publishing, not per commit.
# CANARY_WORK=/some/dir keeps the work dir (and its repo) for debugging instead of
# deleting it on exit. Honors MAVEN / MAVEN_FLAGS like the Makefiles.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")" && pwd)
# unquoted MAVEN_FLAGS: word-splitting intended, same convention as the Makefiles
MVN="${MAVEN:-mvn} ${MAVEN_FLAGS:-}"

WORK=${CANARY_WORK:-$(mktemp -d /tmp/release-canary.XXXXXX)}
if [ -z "${CANARY_WORK:-}" ]; then trap 'rm -rf "$WORK"' EXIT; fi
REPO=$WORK/repo

VERSION=$($MVN -q -B -f "$ROOT/pom.xml" help:evaluate -Dexpression=project.version \
    -DforceStdout 2>/dev/null | tail -1)
case "$VERSION" in
    ''|*' '*) echo "canary: could not determine project version (got '$VERSION')" >&2; exit 1 ;;
esac

echo "==> installing the $VERSION release build into throwaway repo $REPO"
# jam.native.skip: the canary exercises pom/artifact resolution, not the cmake build.
# shellcheck disable=SC2086
$MVN -B -q -f "$ROOT/pom.xml" -Prelease install \
    -pl jinfer/jinfer-bom,jinfer/jinfer-langchain4j -am \
    -DskipTests -Dspotless.check.skip=true -Dgpg.skip=true -Djam.native.skip=true \
    -Dmaven.repo.local="$REPO"

mkdir -p "$WORK/consumer/src/main/java/canary"
cat > "$WORK/consumer/pom.xml" <<EOF
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>canary</groupId>
  <artifactId>release-canary-consumer</artifactId>
  <version>0</version>
  <properties>
    <maven.compiler.release>25</maven.compiler.release>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
  </properties>
  <dependencyManagement>
    <dependencies>
      <dependency>
        <groupId>com.qxotic</groupId>
        <artifactId>jinfer-bom</artifactId>
        <version>$VERSION</version>
        <type>pom</type>
        <scope>import</scope>
      </dependency>
    </dependencies>
  </dependencyManagement>
  <dependencies>
    <!-- no version: managed by the BOM, or this build fails -->
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-langchain4j</artifactId>
    </dependency>
  </dependencies>
</project>
EOF
cat > "$WORK/consumer/src/main/java/canary/Canary.java" <<'EOF'
package canary;

import com.qxotic.jinfer.langchain4j.JinferChatModel;
import java.nio.file.Path;

public class Canary {

    // Compile-only: proves the flattened pom's transitive closure is complete. Never invoked -
    // build() would load a model.
    static JinferChatModel representativeBuilderCall(Path gguf) {
        return JinferChatModel.builder().modelPath(gguf).build();
    }
}
EOF

echo "==> compiling the consumer against ONLY the throwaway repository"
# shellcheck disable=SC2086
( cd "$WORK/consumer" && $MVN -B -q compile -Dmaven.repo.local="$REPO" )

echo "==> canary green: jinfer-bom $VERSION manages jinfer-langchain4j and its flattened poms resolve"
