#!/usr/bin/env bash
# Native canary: prove an application's GraalVM native image of jinfer works with no configuration.
#
#   1. a plain LangChain4j app, built with native-image
#   2. a Spring Boot app on the jinfer starter, built with mvn -Pnative native:compile
#
# Each must answer "Paris", and each build must log jinfer's vector-width pin, without which an image
# runs a hundred times slower. Both resolve com.qxotic from the local repository, so run `make install`
# (or the release install) first, with GraalVM 25 on the PATH. Not part of the default build: it
# downloads the Spring Boot plugins and LFM2.5-350M once, and takes minutes. A failure keeps the work
# dir and its logs; NATIVE_CANARY_WORK=/some/dir keeps it always. Honors MAVEN / MAVEN_FLAGS like the
# Makefiles.

set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
# unquoted MAVEN_FLAGS: word-splitting intended, same convention as the Makefiles
MVN="${MAVEN:-mvn} ${MAVEN_FLAGS:-}"
MODEL=${NATIVE_CANARY_MODEL:-LiquidAI/LFM2.5-350M-GGUF:Q8_0}

command -v native-image >/dev/null || { echo "native canary: native-image is not on the PATH (GraalVM 25)" >&2; exit 1; }
WORK=${NATIVE_CANARY_WORK:-$(mktemp -d "${TMPDIR:-/tmp}/native-canary.XXXXXX")}
fail() { echo "native canary: $1, logs in $WORK" >&2; exit 1; }

evaluate() {
    $MVN -q -B -f "$ROOT/$1/pom.xml" help:evaluate -Dexpression="$2" -DforceStdout 2>/dev/null | tail -1
}
JINFER=$(evaluate jinfer/jinfer-bom project.version)
BOOT=$(evaluate jinfer spring-boot.version)
SPRING_AI=$(evaluate jinfer spring-ai.version)

# An app's build log carries the pin's line, and its run says Paris.
check() {
    local pin
    pin=$(grep -m1 'jinfer: Vector API width' "$1/build.log") || fail "$2 built without the vector-width pin"
    grep -q 'Paris' "$1/run.log" || fail "$2 did not answer Paris"
    echo "==> $2: $pin; answered Paris"
}

BOM="<dependency><groupId>com.qxotic</groupId><artifactId>jinfer-bom</artifactId><version>$JINFER</version><type>pom</type><scope>import</scope></dependency>"
APP_DEPS="<dependency><groupId>com.qxotic</groupId><artifactId>jinfer-lfm2</artifactId></dependency>
    <dependency><groupId>com.qxotic</groupId><artifactId>jam-native</artifactId><scope>runtime</scope></dependency>"

echo "==> LangChain4j app (jinfer $JINFER)"
L=$WORK/langchain4j
mkdir -p "$L/src/main/java/canary"
cat > "$L/pom.xml" <<EOF
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>canary</groupId><artifactId>langchain4j-native</artifactId><version>0</version>
  <properties><maven.compiler.release>25</maven.compiler.release></properties>
  <dependencyManagement><dependencies>$BOM</dependencies></dependencyManagement>
  <dependencies>
    <dependency><groupId>com.qxotic</groupId><artifactId>jinfer-langchain4j</artifactId></dependency>
    $APP_DEPS
  </dependencies>
</project>
EOF
cat > "$L/src/main/java/canary/Main.java" <<'EOF'
package canary;

import com.qxotic.jinfer.langchain4j.JinferChatModel;

public class Main {
    public static void main(String[] args) {
        try (var model = JinferChatModel.builder().model(args[0]).build()) {
            System.out.println(model.chat("What is the capital of France? One word."));
        }
    }
}
EOF
# shellcheck disable=SC2086
( cd "$L" && $MVN -B -q package dependency:build-classpath -Dmdep.outputFile=cp.txt \
    && native-image -cp "target/langchain4j-native-0.jar:$(cat cp.txt)" canary.Main -o target/app > build.log 2>&1 \
    && ./target/app "$MODEL" > run.log 2>&1 ) || fail "the LangChain4j app did not build or run"
check "$L" "LangChain4j"

echo "==> Spring Boot app (Boot $BOOT, Spring AI $SPRING_AI)"
B=$WORK/spring-boot
mkdir -p "$B/src/main/java/canary" "$B/src/main/resources"
cat > "$B/pom.xml" <<EOF
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <parent>
    <groupId>org.springframework.boot</groupId><artifactId>spring-boot-starter-parent</artifactId>
    <version>$BOOT</version><relativePath/>
  </parent>
  <groupId>canary</groupId><artifactId>spring-boot-native</artifactId><version>0</version>
  <properties><java.version>25</java.version></properties>
  <dependencyManagement><dependencies>
    $BOM
    <dependency><groupId>org.springframework.ai</groupId><artifactId>spring-ai-bom</artifactId><version>$SPRING_AI</version><type>pom</type><scope>import</scope></dependency>
  </dependencies></dependencyManagement>
  <dependencies>
    <dependency><groupId>com.qxotic</groupId><artifactId>jinfer-spring-ai-spring-boot-starter</artifactId></dependency>
    $APP_DEPS
  </dependencies>
  <build><plugins>
    <plugin><groupId>org.graalvm.buildtools</groupId><artifactId>native-maven-plugin</artifactId></plugin>
    <plugin><groupId>org.springframework.boot</groupId><artifactId>spring-boot-maven-plugin</artifactId></plugin>
  </plugins></build>
</project>
EOF
cat > "$B/src/main/java/canary/App.java" <<'EOF'
package canary;

import org.springframework.ai.chat.model.ChatModel;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

@SpringBootApplication
public class App {
    public static void main(String[] args) {
        SpringApplication.run(App.class, args);
    }

    @Bean
    CommandLineRunner ask(ChatModel chat) {
        return args -> System.out.println(chat.call("What is the capital of France? One word."));
    }
}
EOF
printf 'spring.main.web-application-type=none\nspring.ai.model.chat=jinfer\n' > "$B/src/main/resources/application.properties"
# shellcheck disable=SC2086
( cd "$B" && $MVN -B -Pnative native:compile > build.log 2>&1 \
    && ./target/spring-boot-native --spring.ai.jinfer.chat.model="$MODEL" > run.log 2>&1 ) || fail "the Spring Boot app did not build or run"
check "$B" "Spring Boot"

[ -n "${NATIVE_CANARY_WORK:-}" ] || rm -rf "$WORK"
echo "==> native canary green: LangChain4j and Spring Boot images build and answer with no configuration"
