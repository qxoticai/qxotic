# Tok'n'Roll JTokkit Adapter

[![Java][badge-java]](https://openjdk.org/projects/jdk/11/)
[![GraalVM Native Image][badge-native-image]](https://www.graalvm.org/latest/reference-manual/native-image/)
[![License][badge-license]](../../LICENSE)

`toknroll-jtokkit` provides the optional JTokkit-backed TikToken adapter for Tok'n'Roll.

This module contains the JTokkit-based TikToken adapter APIs for Tok'n'Roll.

## What it is

- Optional adapter module for `JTokkitTokenizers.fromTiktoken(...)`
- Uses JTokkit under the hood for TikToken-compatible behavior
- Kept separate so core `toknroll` stays minimal

## Dependency

```xml
<dependency>
  <groupId>com.qxotic</groupId>
  <artifactId>toknroll-jtokkit</artifactId>
  <version>0.1.0</version>
</dependency>
```

[badge-java]: https://img.shields.io/badge/Java-11+-blue
[badge-native-image]: https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F
[badge-license]: https://img.shields.io/badge/license-Apache%202.0-green
