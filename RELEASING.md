# Release checklist

Run commands from the repository root with Maven 3.9+ and JDK 25.
Use Oracle GraalVM 25 to exercise both JIT paths below.
Record the commit, platform, commands, test totals and skips with the release evidence.
A skipped test is not a pass.

## Build and test

```sh
make ci
JAVA_TOOL_OPTIONS=-XX:-UseJVMCICompiler make ci-test ci-corpus
```

`make ci` downloads the tokenizer fixtures, checks formatting, runs model-free and corpus tests, and verifies unsigned release packaging.
Formatting and model-free tests include the opt-in Maple and jinfer examples.
The release-packaging step starts with `clean`, so preserve test reports before it if needed.
It skips tests, signatures and native-library verification: a green result does not mean the artifacts are ready to publish.

Run the CI model-contract gate with its small checkpoint:

```sh
jinfer/scripts/download-models.sh --only LFM2.5-350M-Q8_0.gguf
mvn -B -pl jinfer/jinfer-cli,jinfer/jinfer-langchain4j -am test \
  -Dtest=CliIT,ChatEngineModelTest,ChatEngineWeightsOwnershipTest \
  -Dsurefire.failIfNoSpecifiedTests=false -Dsurefire.excludedGroups= \
  -Djinfer.test.requireModels=true
```

This is a smoke gate, not full model coverage.
Run the opt-in suites documented by each affected module, including tokenizer parity, chat and speech integrations, and native backend parity where applicable.
Record missing models, tools and hardware explicitly rather than treating their skips as release evidence.

Check the documentation site with the locked dependencies:

```sh
npm ci
npm run typecheck
npm run build
```

## Consumer artifacts and native libraries

```sh
make release-canary
```

The canary installs release artifacts into a temporary Maven repository and compiles isolated LangChain4j, Spring AI and Spring Boot consumers with and without BOMs.
It requires network access for dependencies and checks dependency resolution and compilation, not inference, signatures or native-library correctness.

Build and test the shipped native libraries using [Building jam](jam/BUILDING.md), then verify the staged set:

```sh
jam/jam-native/scripts/natives.sh verify
mvn -B -Prelease verify -DskipTests -Dgpg.skip=true
```

Do not pass `-Djam.natives.check.skip=true` for this gate.
The second command checks unsigned release packaging with the staged native libraries; it does not replace the tests above.
Smoke-test any native executables intended for distribution on their target platforms and use the portable `compatibility` architecture setting, not the host-specific `make native` default.

## Before publishing

- Review [release notes](RELEASE-NOTES.md), supported model families, API documentation and known limitations.
- Confirm the intended versions and `project.build.outputTimestamp`; do not change them as a side effect of QA.
- A project (gguf, json, safetensors, jota, jam, toknroll, jinfer) is released as a whole, and every artifact in it takes the project's version.
  Between two releases of a project, an artifact whose code changed is released alone, on a version of its own.
  The next release of the project removes those versions again.
- A project's version is its own root POM plus the matching `<project>.version` property (in the repository root POM; for jinfer, in `jinfer/pom.xml`), and its entries in `jinfer-bom`.
  An artifact's own version is three: its POM, the `<artifactId>.version` property next to its project's, and its entry in `jinfer-bom`.
- Run `make release-plan`: it fails when those edits disagree or when code changed under a version Maven Central holds, and it lists what to stage.
- An artifact released alone keeps every public member its last release had: a newer `jinfer-kernels` runs under the model jars compiled against the older one.
- Tag what was published, `<artifactId>-v<version>` or `<project>-v<version>`: the tag is what `make release-plan` compares the code against.
- Inspect the artifacts that opt into publication, including POM dependencies, source and Javadoc JARs, LICENSE and NOTICE files, and native-library contents.
- Run the signing-enabled release verification with the configured release key, without `gpg.skip` or native-check bypasses.
- Resolve failures and document coverage gaps before deciding whether to release.
- Stage only the project whose version moved. Central rejects a coordinate it already holds and one rejection fails the whole bundle, so deploying from the repository root, which restages every artifact that opts into publication, is wrong. Publish one project at a time:

```sh
mvn -Prelease install -DskipTests            # once, so the projects not being released resolve
make release-deploy PROJECT=jinfer CHECK=--check   # what it would stage, and whether Central has it
make release-deploy PROJECT=jinfer
```

It builds only that project's reactor and refuses before uploading anything if one of its coordinates is already published.

Publishing and tagging are separate, explicit maintainer actions.
The release profile leaves publication approval manual in Central (`autoPublish=false`); `deploy` still uploads artifacts and must not be used as a local QA check.
