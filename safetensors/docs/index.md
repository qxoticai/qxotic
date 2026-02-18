# Safetensors

A Java library for reading and writing [Safetensors](https://github.com/huggingface/safetensors) headers.

## What is covered

This library works on the header section:

- `__metadata__` map (string keys/values)
- tensor descriptors (`dtype`, `shape`, `data_offsets`)
- alignment metadata (`__alignment__`)

It does not load tensor payload bytes automatically.

## Installation

=== "Maven"

    ```xml
    <dependency>
        <groupId>ai.qxotic</groupId>
        <artifactId>safetensors</artifactId>
        <version>0.1-SNAPSHOT</version>
    </dependency>
    ```

=== "Gradle"

    ```groovy
    implementation 'ai.qxotic:safetensors:0.1-SNAPSHOT'
    ```

## Reading

### From a file

```java
--8<-- "src/test/java/ai/qxotic/format/safetensors/Snippets.java:read-path"
```

### From a byte channel

```java
--8<-- "src/test/java/ai/qxotic/format/safetensors/Snippets.java:read-channel"
```

### From Hugging Face URL

```java
--8<-- "src/test/java/ai/qxotic/format/safetensors/Snippets.java:read-url"
```

### Reusable helper for HF repos

```java
--8<-- "src/test/java/ai/qxotic/format/safetensors/Snippets.java:read-from-huggingface"
```

## Inspecting header data

### Basic fields

```java
--8<-- "src/test/java/ai/qxotic/format/safetensors/Snippets.java:basic-info"
```

### Metadata

```java
--8<-- "src/test/java/ai/qxotic/format/safetensors/Snippets.java:metadata"
```

### Tensors

```java
--8<-- "src/test/java/ai/qxotic/format/safetensors/Snippets.java:tensors"
```

### One tensor

```java
--8<-- "src/test/java/ai/qxotic/format/safetensors/Snippets.java:tensor-one"
```

## Writing

### Build new header

```java
--8<-- "src/test/java/ai/qxotic/format/safetensors/Snippets.java:builder-create"
```

### Modify existing header

```java
--8<-- "src/test/java/ai/qxotic/format/safetensors/Snippets.java:builder-modify"
```

### Control alignment

```java
--8<-- "src/test/java/ai/qxotic/format/safetensors/Snippets.java:builder-alignment"
```

### Write to file

```java
--8<-- "src/test/java/ai/qxotic/format/safetensors/Snippets.java:write-file"
```

## Sharded models

Load directory index and resolve shard file for each tensor:

```java
--8<-- "src/test/java/ai/qxotic/format/safetensors/Snippets.java:index-load"
```

## Validation behavior

The parser is strict and throws `SafetensorsFormatException` for:

- invalid JSON header/root shape
- missing tensor fields
- unsupported dtype
- non-integer/out-of-range shape or offset values
- invalid/overlapping offsets
- invalid `__alignment__` metadata

Duplicate JSON keys follow the underlying JSON parser policy (last key wins).

## CLI quick inspect with JBang

```bash
jbang scripts/safetensors.java hf HuggingFaceTB/SmolLM2-135M --no-tensors
jbang scripts/safetensors.java modelscope Qwen/Qwen3-4B --no-tensors
```

The script emits pure JSON on stdout.
