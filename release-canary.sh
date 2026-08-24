#!/usr/bin/env bash
# Release canary: prove the artifacts a consumer would actually GET are usable.
#
#   1. install the release build into a throwaway local repository
#   2. compile isolated LangChain4j, Spring AI, and Spring Boot consumers against it
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
LANGCHAIN4J_VERSION=$($MVN -q -B -f "$ROOT/jinfer/jinfer-langchain4j/pom.xml" help:evaluate \
    -Dexpression=langchain4j.version -DforceStdout 2>/dev/null | tail -1)
SPRING_AI_VERSION=$($MVN -q -B -f "$ROOT/jinfer/pom.xml" help:evaluate \
    -Dexpression=spring-ai.version -DforceStdout 2>/dev/null | tail -1)
case "$VERSION" in
    ''|*' '*) echo "canary: could not determine project version (got '$VERSION')" >&2; exit 1 ;;
esac
case "$LANGCHAIN4J_VERSION" in
    ''|*' '*) echo "canary: could not determine LangChain4j version" >&2; exit 1 ;;
esac
case "$SPRING_AI_VERSION" in
    ''|*' '*) echo "canary: could not determine Spring AI version" >&2; exit 1 ;;
esac

echo "==> installing the $VERSION release build into throwaway repo $REPO"
# jam.native.skip: the canary exercises pom/artifact resolution, not the cmake build.
# shellcheck disable=SC2086
$MVN -B -q -f "$ROOT/pom.xml" -Prelease install \
    -pl jinfer/jinfer-bom,jinfer/jinfer-langchain4j,jinfer/jinfer-spring-ai,jinfer/jinfer-spring-ai-autoconfigure,jinfer/jinfer-spring-ai-spring-boot-starter,jinfer/jinfer-lfm2,jinfer/jinfer-models-all -am \
    -DskipTests -Dspotless.check.skip=true -Dgpg.skip=true -Djam.native.skip=true \
    -Dmaven.repo.local="$REPO"

mkdir -p \
    "$WORK/consumer/langchain4j-core/src/main/java/canary" \
    "$WORK/consumer/langchain4j-ai-services/src/main/java/canary" \
    "$WORK/consumer/spring-ai-core/src/main/java/canary" \
    "$WORK/consumer/spring-ai-boot/src/main/java/canary" \
    "$WORK/consumer/models-all"

cat > "$WORK/consumer/pom.xml" <<EOF
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>canary</groupId>
  <artifactId>release-canary-consumer</artifactId>
  <version>0</version>
  <packaging>pom</packaging>
  <modules>
    <module>langchain4j-core</module>
    <module>langchain4j-ai-services</module>
    <module>spring-ai-core</module>
    <module>spring-ai-boot</module>
    <module>models-all</module>
  </modules>
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
      <dependency>
        <groupId>dev.langchain4j</groupId>
        <artifactId>langchain4j-bom</artifactId>
        <version>$LANGCHAIN4J_VERSION</version>
        <type>pom</type>
        <scope>import</scope>
      </dependency>
      <dependency>
        <groupId>org.springframework.ai</groupId>
        <artifactId>spring-ai-bom</artifactId>
        <version>$SPRING_AI_VERSION</version>
        <type>pom</type>
        <scope>import</scope>
      </dependency>
    </dependencies>
  </dependencyManagement>
</project>
EOF

cat > "$WORK/consumer/langchain4j-core/pom.xml" <<'EOF'
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>canary</groupId>
    <artifactId>release-canary-consumer</artifactId>
    <version>0</version>
  </parent>
  <artifactId>langchain4j-core-canary</artifactId>
  <dependencies>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-langchain4j</artifactId>
    </dependency>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-lfm2</artifactId>
    </dependency>
  </dependencies>
</project>
EOF
cat > "$WORK/consumer/langchain4j-core/src/main/java/canary/Canary.java" <<'EOF'
package canary;

import com.qxotic.jinfer.langchain4j.JinferChatModel;
import java.nio.file.Path;

final class Canary {

    static JinferChatModel create(Path model) {
        return JinferChatModel.builder().modelPath(model).build();
    }
}
EOF

cat > "$WORK/consumer/langchain4j-ai-services/pom.xml" <<'EOF'
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>canary</groupId>
    <artifactId>release-canary-consumer</artifactId>
    <version>0</version>
  </parent>
  <artifactId>langchain4j-ai-services-canary</artifactId>
  <dependencies>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-langchain4j</artifactId>
    </dependency>
    <dependency>
      <groupId>dev.langchain4j</groupId>
      <artifactId>langchain4j</artifactId>
    </dependency>
  </dependencies>
</project>
EOF
cat > "$WORK/consumer/langchain4j-ai-services/src/main/java/canary/Canary.java" <<'EOF'
package canary;

import com.qxotic.jinfer.langchain4j.JinferChatModel;
import dev.langchain4j.service.AiServices;

final class Canary {

    interface Assistant {
        String chat(String message);
    }

    static Assistant create(JinferChatModel model) {
        return AiServices.create(Assistant.class, model);
    }
}
EOF

cat > "$WORK/consumer/spring-ai-core/pom.xml" <<'EOF'
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>canary</groupId>
    <artifactId>release-canary-consumer</artifactId>
    <version>0</version>
  </parent>
  <artifactId>spring-ai-core-canary</artifactId>
  <dependencies>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-spring-ai</artifactId>
    </dependency>
  </dependencies>
</project>
EOF
cat > "$WORK/consumer/spring-ai-core/src/main/java/canary/Canary.java" <<'EOF'
package canary;

import com.qxotic.jinfer.spring.ai.JinferChatModel;
import java.nio.file.Path;
import org.springframework.ai.chat.model.ChatModel;

final class Canary {

    static ChatModel create(Path model) {
        return JinferChatModel.builder().modelPath(model).build();
    }
}
EOF

cat > "$WORK/consumer/spring-ai-boot/pom.xml" <<'EOF'
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>canary</groupId>
    <artifactId>release-canary-consumer</artifactId>
    <version>0</version>
  </parent>
  <artifactId>spring-ai-boot-canary</artifactId>
  <dependencies>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-spring-ai-spring-boot-starter</artifactId>
    </dependency>
  </dependencies>
</project>
EOF
cat > "$WORK/consumer/spring-ai-boot/src/main/java/canary/Canary.java" <<'EOF'
package canary;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.boot.autoconfigure.SpringBootApplication;

@SpringBootApplication
class Canary {

    private final ChatClient chat;

    Canary(ChatClient.Builder builder) {
        this.chat = builder.build();
    }
}
EOF

cat > "$WORK/consumer/models-all/pom.xml" <<'EOF'
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>canary</groupId>
    <artifactId>release-canary-consumer</artifactId>
    <version>0</version>
  </parent>
  <artifactId>models-all-canary</artifactId>
  <dependencies>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-models-all</artifactId>
    </dependency>
  </dependencies>
</project>
EOF

echo "==> compiling isolated consumers against only the throwaway repository"
# shellcheck disable=SC2086
( cd "$WORK/consumer" && $MVN -B -q compile -Dmaven.repo.local="$REPO" )

echo "==> canary green: published BOMs and integration POMs resolve for all supported entry points"
