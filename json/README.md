# JSON

[![Java](https://img.shields.io/badge/Java-11+-blue)](https://openjdk.org/projects/jdk/11/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

Strict JSON parser/writer for Java. Parses JSON directly into standard Java collections—zero dependencies, RFC 8259 compliant.

## Overview

| JSON     | Java Type                     |
|----------|-------------------------------|
| Objects  | `Map<String, Object>`         |
| Arrays   | `List<Object>`                |
| Strings  | `String`                      |
| Numbers  | `Long`, `BigInteger`, `BigDecimal` |
| Booleans | `Boolean`                     |
| null     | `Json.NULL`                   |

**Why use this?**
- No POJO classes to define
- No reflection or bytecode generation  
- Single class (~1000 lines), zero dependencies
- GraalVM Native Image compatible

## Quick Start

```xml
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>json</artifactId>
    <version>0.1.0</version>
</dependency>
```

```java
// Parse JSON
Map<String, Object> person = Json.parseMap("{\"name\":\"alice\",\"age\":30}");
String name = (String) person.get("name");

// Query nested data
Optional<String> city = Json.queryString(person, "address", "city");

// Serialize
String json = Json.stringify(person);
```

## API

```java
// Parsing
Map<String, Object> obj = Json.parseMap(json);
List<Object> arr = Json.parseList(json);
Optional<String> str = Json.queryString(root, "path", "to", "value");
boolean valid = Json.isValid(json);

// Serialization
String compact = Json.stringify(data);
String pretty = Json.stringify(data, true);

// Configuration
Json.ParseOptions opts = Json.ParseOptions.defaults()
    .maxDepth(50)
    .failOnDuplicateKeys(true);
```

## When to Use

**Good for:** Configuration files, REST clients, microservices (~10KB vs Jackson ~3MB), GraalVM native images

**Not for:** Heavy POJO mapping, streaming large files, complex serialization

## Benchmarks

```bash
cd benchmarks && mvn clean package && java -jar target/json-benchmarks.jar
```

| Operation | Qxotic | Jackson | Difference |
|-----------|--------|---------|------------|
| Parse Small (~500B) | 0.81 µs | 0.88 µs | ~8% faster |
| Round-trip Small | 1.36 µs | 1.70 µs | ~20% faster |

See [benchmarks/](benchmarks/) for details.

## License

Apache License 2.0
