#!/usr/bin/env bash
# Release canary: prove the artifacts a consumer would actually GET are usable.
#
#   1. install the release build into a throwaway local repository
#   2. compile isolated LangChain4j, Spring AI, and Spring Boot consumers against it - chat and
#      speech (Kokoro, with its voice companion) through both frameworks' own model types
#   3. compile the same entry points with explicit versions and no imported BOMs
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

WORK=${CANARY_WORK:-$(mktemp -d "${TMPDIR:-/tmp}/release-canary.XXXXXX")}
if [ -z "${CANARY_WORK:-}" ]; then trap 'rm -rf "$WORK"' EXIT; fi
REPO=$WORK/repo

# Expression $2 of the POM in directory $1, as the build sees it.
evaluate() {
    $MVN -q -B -f "$ROOT/$1/pom.xml" help:evaluate \
        -Dexpression="$2" -DforceStdout 2>/dev/null | tail -1
}
# An artifact released alone has a version of its own, so each consumer asks for that one.
version_of() { evaluate "$1" project.version; }
# The catalog can have a version of its own.
BOM_VERSION=$(version_of jinfer/jinfer-bom)
LANGCHAIN4J_VERSION=$(evaluate jinfer langchain4j.version)
SPRING_AI_VERSION=$(evaluate jinfer spring-ai.version)
case "$BOM_VERSION" in
    ''|*' '*) echo "canary: could not determine the BOM version (got '$BOM_VERSION')" >&2; exit 1 ;;
esac
case "$LANGCHAIN4J_VERSION" in
    ''|*' '*) echo "canary: could not determine LangChain4j version" >&2; exit 1 ;;
esac
case "$SPRING_AI_VERSION" in
    ''|*' '*) echo "canary: could not determine Spring AI version" >&2; exit 1 ;;
esac

echo "==> installing the release build (catalog $BOM_VERSION) into throwaway repo $REPO"
# jam.natives.check.skip: the canary exercises pom/artifact resolution, not the native libraries
# (the release profile itself never runs cmake; it packages the staged, verified set).
# shellcheck disable=SC2086
$MVN -B -q -f "$ROOT/pom.xml" -Prelease install \
    -pl jinfer/jinfer-bom,jinfer/jinfer-langchain4j,jinfer/jinfer-spring-ai,jinfer/jinfer-spring-ai-autoconfigure,jinfer/jinfer-spring-ai-spring-boot-starter,jinfer/jinfer-decision,jinfer/jinfer-lfm2,jinfer/jinfer-models-all -am \
    -DskipTests -Dspotless.check.skip=true -Dgpg.skip=true -Djam.natives.check.skip=true \
    -Dmaven.repo.local="$REPO"

mkdir -p \
    "$WORK/consumer/langchain4j-core/src/main/java/canary" \
    "$WORK/consumer/langchain4j-ai-services/src/main/java/canary" \
    "$WORK/consumer/spring-ai-core/src/main/java/canary" \
    "$WORK/consumer/spring-ai-boot/src/main/java/canary" \
    "$WORK/consumer/models-all" \
    "$WORK/no-bom-providers/src/main/java/canary" \
    "$WORK/no-bom-boot/src/main/java/canary"

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
        <version>$BOM_VERSION</version>
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
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-kokoro</artifactId>
    </dependency>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-decision</artifactId>
    </dependency>
  </dependencies>
</project>
EOF
cat > "$WORK/consumer/langchain4j-core/src/main/java/canary/Canary.java" <<'EOF'
package canary;

import com.qxotic.jinfer.ContextState;
import com.qxotic.jinfer.decision.Answer;
import com.qxotic.jinfer.decision.Decisions;
import com.qxotic.jinfer.decision.Question;
import com.qxotic.jinfer.langchain4j.JinferChatModel;
import com.qxotic.jinfer.langchain4j.JinferDecisionModel;
import com.qxotic.jinfer.langchain4j.JinferSpeechModel;
import dev.langchain4j.model.audio.TextToSpeechModel;
import dev.langchain4j.model.decision.DecisionModel;
import dev.langchain4j.model.decision.request.ChoiceQuestion;
import dev.langchain4j.model.decision.request.DecisionRequest;
import dev.langchain4j.model.decision.request.ScaleQuestion;
import dev.langchain4j.model.decision.request.YesNoQuestion;
import dev.langchain4j.model.decision.response.DecisionResponse;
import java.nio.file.Path;
import java.util.List;

final class Canary {

    static JinferChatModel create(Path model) {
        return JinferChatModel.builder().modelPath(model).build();
    }

    static TextToSpeechModel speech(Path model, Path voice) {
        return JinferSpeechModel.builder().modelPath(model).companionPath("voice", voice).build();
    }

    /** The framework face: a decision model answers typed questions, its media companion attached. */
    static DecisionResponse decide(DecisionModel model) {
        return model.decide(
                DecisionRequest.builder()
                        .input("I was charged twice and I want my money back today.")
                        .question("refund", YesNoQuestion.from("Is this a refund request?"))
                        .question(
                                "team",
                                ChoiceQuestion.from(
                                        "Which team?", List.of("billing", "shipping", "other")))
                        .question(
                                "urgency",
                                ScaleQuestion.from("How urgent?", List.of("Can wait", "Today")))
                        .build());
    }

    static DecisionModel decision(Path model, Path media) {
        return JinferDecisionModel.builder()
                .modelPath(model)
                .companionPath("media", media)
                .build();
    }

    /** The typed face over the engine, as documented in jinfer's README. */
    static <T extends Enum<T>> T route(
            Decisions decisions, ContextState state, Question.Choice<T> question) {
        return decisions.ask(state, "I was charged twice and I want my money back today.", question)
                .selectedOption();
    }

    static boolean refund(Decisions decisions, ContextState state) {
        Answer.Predicate answer =
                decisions.ask(
                        state,
                        "I was charged twice and I want my money back today.",
                        Question.predicate("Is this a refund request?"));
        return answer.isYes();
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
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-kokoro</artifactId>
    </dependency>
  </dependencies>
</project>
EOF
cat > "$WORK/consumer/spring-ai-core/src/main/java/canary/Canary.java" <<'EOF'
package canary;

import com.qxotic.jinfer.spring.ai.JinferChatModel;
import com.qxotic.jinfer.spring.ai.JinferDecisionModel;
import com.qxotic.jinfer.spring.ai.JinferSpeechModel;
import java.nio.file.Path;
import java.util.Map;
import org.springframework.ai.audio.tts.TextToSpeechModel;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.decision.DecisionModel;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.input.DecisionInput;
import org.springframework.ai.decision.question.ChoiceQuestion;
import org.springframework.ai.decision.question.ScoreQuestion;
import org.springframework.ai.decision.question.YesNoQuestion;

final class Canary {

    static ChatModel create(Path model) {
        return JinferChatModel.builder().modelPath(model).build();
    }

    static TextToSpeechModel speech(Path model, Path voice) {
        return JinferSpeechModel.builder().modelPath(model).companionPath("voice", voice).build();
    }

    /** The framework face: fields in, one typed answer per question named. */
    static DecisionResponse decide(DecisionModel model) {
        return model.call(
                DecisionInput.fields(Map.of("message", "I was charged twice.")),
                Map.of(
                        "refund", YesNoQuestion.of("Is this a refund request?"),
                        "team", ChoiceQuestion.of("Which team?", "billing", "shipping", "other"),
                        "urgency", ScoreQuestion.of("How urgent?", "Can wait", "Today")));
    }

    static DecisionModel decision(Path model, Path media) {
        return JinferDecisionModel.builder()
                .modelPath(model)
                .companionPath("media", media)
                .together(true)
                .build();
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

cat > "$WORK/no-bom-providers/pom.xml" <<EOF
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>canary</groupId>
  <artifactId>no-bom-providers-canary</artifactId>
  <version>0</version>
  <properties>
    <maven.compiler.release>25</maven.compiler.release>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
  </properties>
  <dependencies>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-langchain4j</artifactId>
      <version>$(version_of jinfer/jinfer-langchain4j)</version>
    </dependency>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-spring-ai</artifactId>
      <version>$(version_of jinfer/jinfer-spring-ai)</version>
    </dependency>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-kokoro</artifactId>
      <version>$(version_of jinfer/jinfer-kokoro)</version>
    </dependency>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-decision</artifactId>
      <version>$(version_of jinfer/jinfer-decision)</version>
    </dependency>
  </dependencies>
</project>
EOF
cat > "$WORK/no-bom-providers/src/main/java/canary/Canary.java" <<'EOF'
package canary;

import java.nio.file.Path;

final class Canary {

    static dev.langchain4j.model.chat.ChatModel langChain4j(Path model) {
        return com.qxotic.jinfer.langchain4j.JinferChatModel.builder().modelPath(model).build();
    }

    static org.springframework.ai.chat.model.ChatModel springAi(Path model) {
        return com.qxotic.jinfer.spring.ai.JinferChatModel.builder().modelPath(model).build();
    }

    static dev.langchain4j.model.audio.TextToSpeechModel langChain4jSpeech(Path model, Path voice) {
        return com.qxotic.jinfer.langchain4j.JinferSpeechModel.builder()
                .modelPath(model).companionPath("voice", voice).build();
    }

    static org.springframework.ai.audio.tts.TextToSpeechModel springAiSpeech(Path model, Path voice) {
        return com.qxotic.jinfer.spring.ai.JinferSpeechModel.builder()
                .modelPath(model).companionPath("voice", voice).build();
    }

    static dev.langchain4j.model.decision.DecisionModel langChain4jDecision(Path model) {
        return com.qxotic.jinfer.langchain4j.JinferDecisionModel.builder().modelPath(model).build();
    }

    static org.springframework.ai.decision.DecisionModel springAiDecision(Path model) {
        return com.qxotic.jinfer.spring.ai.JinferDecisionModel.builder().modelPath(model).build();
    }

    static com.qxotic.jinfer.decision.Decisions decisions(
            com.qxotic.jinfer.chat.DecisionModel model) {
        return com.qxotic.jinfer.decision.Decisions.over(model, 2048);
    }
}
EOF

cat > "$WORK/no-bom-boot/pom.xml" <<EOF
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>canary</groupId>
  <artifactId>no-bom-boot-canary</artifactId>
  <version>0</version>
  <properties>
    <maven.compiler.release>25</maven.compiler.release>
    <project.build.sourceEncoding>UTF-8</project.build.sourceEncoding>
  </properties>
  <dependencies>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-spring-ai-spring-boot-starter</artifactId>
      <version>$(version_of jinfer/jinfer-spring-ai-spring-boot-starter)</version>
    </dependency>
    <dependency>
      <groupId>com.qxotic</groupId>
      <artifactId>jinfer-models-all</artifactId>
      <version>$(version_of jinfer/jinfer-models-all)</version>
    </dependency>
  </dependencies>
</project>
EOF
cat > "$WORK/no-bom-boot/src/main/java/canary/Canary.java" <<'EOF'
package canary;

final class Canary {

    static org.springframework.ai.chat.client.ChatClient chatClient(
            org.springframework.ai.chat.client.ChatClient.Builder builder) {
        return builder.build();
    }
}
EOF

echo "==> compiling isolated consumers against only the throwaway repository"
# shellcheck disable=SC2086
( cd "$WORK/consumer" && $MVN -B -q compile -Dmaven.repo.local="$REPO" )
# shellcheck disable=SC2086
( cd "$WORK/no-bom-providers" && $MVN -B -q compile -Dmaven.repo.local="$REPO" )
# shellcheck disable=SC2086
( cd "$WORK/no-bom-boot" && $MVN -B -q compile -Dmaven.repo.local="$REPO" )

# A com.qxotic version this build does not produce is fetched from Central, and the consumers
# compile against the last release. The resolver marks what it fetched: ">central=", not ">=".
FETCHED=$({ grep -rlE '>[^=]+=' --include=_remote.repositories "$REPO/com/qxotic" 2>/dev/null || true; } |
    sed "s|^$REPO/com/qxotic/||; s|/_remote.repositories$||; s|/|:|" | sort)
if [ -n "$FETCHED" ]; then
    echo "canary: resolved from a remote, not from this build:" $FETCHED >&2
    echo "canary: a POM or the catalog names a version the reactor does not build" >&2
    exit 1
fi

echo "==> canary green: published integration POMs resolve with and without BOMs"
