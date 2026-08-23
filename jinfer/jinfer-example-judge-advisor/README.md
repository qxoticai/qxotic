# Local judge advisor with Spring AI

[![Java 25+](https://img.shields.io/badge/Java-25%2B-007396?logo=java&logoColor=white)](https://openjdk.org/projects/jdk/25/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg?logo=apache)](../LICENSE)

This example pairs a remote generator with a local evaluator. Kimi produces an answer, a Jinfer
model evaluates it in the application JVM, and a Spring AI advisor retries with the evaluator's
feedback until the answer passes or the attempt limit is reached.

It adapts Spring AI's
[`evaluation-recursive-advisor-demo`](https://github.com/spring-projects/spring-ai-examples/tree/main/advisors/evaluation-recursive-advisor-demo)
and adds three safeguards:

1. The verdict schema is compiled to GBNF and enforced during sampling.
2. The rating schema accepts only the integers 1 through 4.
3. The evaluator rubric is prefilled once and restored from the prompt cache for each attempt.

## Example run

```text
>>> tool: weather(Paris) -> -255C
>>> judge: attempt 1 failed (rating 1, 249 tokens, cacheRead 163, decode PT6.7S): The response
    is factually wrong... [answer: The current weather in Paris is sunny with a temperature of -255°C.]
>>> tool: weather(Paris) -> 15C
>>> judge: attempt 2 passed (rating 4, 233 tokens, cacheRead 163, decode PT5.6S)
    [answer: The current weather in Paris is sunny with a temperature of 15C.]
>>> FINAL: The current weather in Paris is sunny with a temperature of 15C.
```

The weather tool returns `-255C` on alternating calls. The first result is intentionally invalid,
which makes the retry path visible.

## Run

From the repository root, build the example and its reactor dependencies:

```bash
mvn -Pexamples -pl jinfer/jinfer-example-judge-advisor -am install -DskipTests
```

Then run the application from this directory:

```bash
export KIMI_API_KEY=sk-kimi-...
export JINFER_JUDGE_MODEL=hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0
mvn spring-boot:run
```

The defaults are `https://api.kimi.com/coding/v1` and `kimi-for-coding`. Override them with
`KIMI_BASE_URL` and `KIMI_MODEL`. Moonshot API keys use `https://api.moonshot.ai/v1` and a model
available to that account. The evaluator model downloads on first use and remains in the Jinfer
cache.

## Design notes

- Reasoning remains enabled because disabling it reduced evaluator accuracy in this test case.
- The rubric permits an explicit statement that real-time data is unavailable.
- Prompt caching removes repeated rubric prefill; verdict decoding remains the main latency cost.
- An 8B evaluator handles this clear factual error. More ambiguous criteria may require a larger
  model.

## Tests

- Unit tests: `mvn test`
- Model-backed tests: `mvn test -Dsurefire.excludedGroups= -Dgroups=integration`
- Kimi tests: run the model-backed command with `KIMI_API_KEY` set

Unit tests use stubs and require neither model files nor an API key. Kimi tests skip when
`KIMI_API_KEY` is absent.
