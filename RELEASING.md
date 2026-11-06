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

## Python wheels

The two wheels, `toknroll` and `jinfer`, are built on each platform's own machine, from a clean export of the release tag.
Linux builds in the manylinux container, so a Linux machine of either architecture builds the Linux wheels of its own; a Mac with docker (Colima) builds Linux aarch64 too.

```sh
make wheels REF=<tag>          # Linux (in the container) or macOS -> target/wheels/
make wheels-linux REF=<tag>    # on a Mac: the Linux aarch64 wheels, in the container
make publish-wheels            # twine check, then upload target/wheels/, skipping wheels PyPI holds
```

On Windows, `.\wheels.ps1 -Ref <tag>` builds them and `.\wheels.ps1 -Publish` uploads them.
Each machine uploads its own wheels; copying every machine's `target/wheels/` to one machine and publishing from there works as well.
Every build runs the libraries' `check-wheel`: the wheel installed into a fresh venv and its suite run from outside the tree.
The suites load real models from the machine's cache, offline, and skip the checks of a model not there; the C suites and `test_jinfer.py` fail the run on a skip unless `CTEST_ALLOW_SKIP=1`, so a machine without the cache cannot look covered.
Cache them once per build machine, online, after both libraries are built there, and put espeak-ng on the PATH for the speech checks (the container image has it):

```sh
make cache-models                  # on Windows: python jinfer\scripts\cache_native_models.py
make cache-models-in-container     # the manylinux container keeps a cache of its own
```

Record the platforms built, their test totals and skips, and the wheel file names with the release evidence.

## C libraries

The native libraries for C are the release archives, one per platform, assets of the GitHub release on the tag: `libtoknroll-<version>-<platform>.tar.gz` and `libjinfer-<version>-<platform>.tar.gz`, a `.zip` on Windows, each a prefix tree (`include/`, `lib/` with the library under its soname, the linker's name, the static archive where there is one, and `lib/pkgconfig/<name>.pc`; `bin\` for the DLL on Windows) with the README and the license.
Each machine builds and checks its own from a clean export of the tag, as the wheels, and they collect in `target/dists/`:

```sh
make dists REF=<tag>           # Linux (in the container) or macOS -> target/dists/
make dists-linux REF=<tag>     # on a Mac: the Linux aarch64 archives, in the container
```

On Windows, `.\dists.ps1 -Ref <tag>` builds the two zips into `target\dists\`.
Copying every machine's `target/dists/` to one machine gives the eight archives and their checksums; the release is created once, `gh release create <tag>`, and the archives uploaded with `gh release upload <tag> target/dists/*`.

## Consumer artifacts and native libraries

```sh
make release-canary
```

The canary installs release artifacts into a temporary Maven repository and compiles isolated LangChain4j, Spring AI and Spring Boot consumers with and without BOMs.
It requires network access for dependencies and checks dependency resolution and compilation, not inference, signatures or native-library correctness.

Build and test the shipped native libraries using [Building jam](jam/BUILDING.md): `make jam-natives` on the Linux host and on the Mac (which also builds the Linux aarch64 library in its container), at the release commit, then the Mac's libraries imported on the Linux host and the staged set verified:

```sh
jam/jam-native/scripts/natives.sh import <the Mac's darwin-aarch64 directory> <its linux-aarch64 directory>
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
- An artifact that depends on a bumped artifact is bumped too, transitively, so no published POM names a stale version of a sibling.
- A project's version is its own root POM plus the matching `<project>.version` property (in the repository root POM; for jinfer, in `jinfer/pom.xml`), and its entries in `jinfer-bom`.
  An artifact's own version is three: its POM, the `<artifactId>.version` property next to its project's, and its entry in `jinfer-bom`.
- Run `make release-plan`: it fails when those edits disagree or when code changed under a version Maven Central holds, and it lists what to stage.
- An artifact released alone keeps every public member its last release had: a newer `jinfer-kernels` runs under the model jars compiled against the older one.
- Tag what was published, `<artifactId>-v<version>` or `<project>-v<version>`: the tag is what `make release-plan` compares the code against.
- Inspect the artifacts that opt into publication, including POM dependencies, source and Javadoc JARs, LICENSE and NOTICE files, and native-library contents.
- Run the signing-enabled release verification with the configured release key, without `gpg.skip` or native-check bypasses.
- Resolve failures and document coverage gaps before deciding whether to release.
- Stage only what Central does not hold yet. Central rejects a coordinate it already holds and one rejection fails the whole bundle, so a plain `deploy` from the repository root, which restages every artifact that opts into publication, is wrong.
  Release everything pending in one bundle, or one project at a time:

```sh
mvn -Prelease install -DskipTests                  # once, so the modules not being staged resolve
make release-deploy PROJECT=. CHECK=--check        # what it would stage, and whether Central has it
make release-deploy PROJECT=.                      # every pending artifact, one deployment to approve
make release-deploy PROJECT=jinfer                 # or one project's reactor
```

From the repository root it leaves out the modules Central already holds (`./release-plan.sh --held`, which also refuses when the plan reports problems), and each project keeps its own version.
For one project it builds only that project's reactor and refuses before uploading anything if one of its coordinates is already published.

Publishing and tagging are separate, explicit maintainer actions.
The release profile leaves publication approval manual in Central (`autoPublish=false`); `deploy` still uploads artifacts and must not be used as a local QA check.
