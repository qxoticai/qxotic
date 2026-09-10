# GGUF

**GGUF for the JVM.**  

Read and write access for [llama.cpp's GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) model format using pure Java.  
Zero dependencies, Java 11+, GraalVM's Native Image ready.

[![Maven Central](https://img.shields.io/maven-central/v/com.qxotic/gguf)](https://search.maven.org/artifact/com.qxotic/gguf)
[![Java](https://img.shields.io/badge/Java-11+-blue)](https://openjdk.org/projects/jdk/11/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![GraalVM](https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F)](https://www.graalvm.org/latest/reference-manual/native-image/)

## Peek inside a GGUF model without downloading it

The example [JBang script](scripts/gguf.java) reads GGUF metadata from HuggingFace, or any arbitrary URL, and
never pulls GBs of tensor data:

```bash
jbang scripts/gguf.java hf unsloth/Qwen3-0.6B-GGUF/Q8_0 --no-tensors
```

## From Java

```java
// Reading
GGUF gguf = GGUF.read(Path.of("model.gguf"));
String name = gguf.getValue(String.class, "general.name");

// Writing
GGUF modified = Builder.newBuilder(gguf)
    .putString("general.description", "My model")
    .build();
GGUF.write(modified, Path.of("output.gguf"));
```

## Maven

```xml
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>gguf</artifactId>
    <version>0.2.0</version>
</dependency>
```

## Deliberately out of scope

- **No tensor payload I/O.** Provide only the tensors offsets and metadata, not reading, loading, nor memory-mapping.
- **No quantization.** Raw bytes only.
- **No inference.** That is [jinfer](../jinfer).
