# GGUF

A Java library for reading and writing [GGUF](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md) files.

## Installation

### Maven

```xml
<dependency>
    <groupId>ai.qxotic</groupId>
    <artifactId>gguf</artifactId>
    <version>0.1.0</version>
</dependency>
```

### Gradle

```groovy
implementation 'ai.qxotic:gguf:0.1.0'
```

### SBT

```scala
libraryDependencies += "ai.qxotic" % "gguf" % "0.1.0"
```

### Mill

```scala
ivy"ai.qxotic::gguf:0.1.0"
```

## Quick Example

```java
import ai.qxotic.format.gguf.*;

// Read a GGUF file
GGUF gguf = GGUF.read(Path.of("model.gguf"));

// Access metadata
String name = gguf.getValue(String.class, "general.name");
int contextLength = gguf.getValue(int.class, "llama.context_length");

// List tensors
for (TensorEntry tensor : gguf.getTensors()) {
    System.out.println(tensor.name() + ": " + tensor.ggmlType());
}
```

## Quick Inspect with JBang

Inspect GGUF metadata from a URL without downloading the full file:

```bash
jbang scripts/gguf.java unsloth/Qwen3-0.6B-GGUF/Q8_0 --no-tensors
```

## Documentation

See [docs/DOCUMENTATION.md](docs/DOCUMENTATION.md) for the full documentation.

## Development

### Building

```bash
mvnd compile
```

### Running Tests

```bash
mvnd test
```

### Building Documentation

Documentation uses MkDocs Material theme. To preview locally:

```bash
pip install mkdocs-material
cd gguf
mkdocs serve
```

### Code Snippets

Code snippets are embedded in `Snippets.java` using region markers (`// region name` / `// endregion name`) and included via `pymdownx.snippets`.

## License

Apache License 2.0
