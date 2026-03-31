# JSON

[![Maven Central](https://img.shields.io/maven-central/v/com.qxotic/json)](https://search.maven.org/artifact/com.qxotic/json)
[![Java](https://img.shields.io/badge/Java-11+-blue)](https://openjdk.org/projects/jdk/11/)
[![License][badge-license]](LICENSE)
[![GraalVM Native Image][badge-native-image]](https://www.graalvm.org/latest/reference-manual/native-image/)

Strict, minimal JSON parser and writer for Java. Parses JSON directly into standard Java collections—**zero dependencies**, **RFC 8259 compliant**.

```java
// Parse JSON in one line
Map<String, Object> data = Json.parseMap("{\"name\":\"alice\",\"age\":30}");

// Serialize back to JSON
String json = Json.stringify(data);
```

## Why Quixotic JSON?

| Feature | Quixotic | Jackson | Gson |
|---------|----------|---------|------|
| Dependencies | **0** | 3+ | 1+ |
| JAR Size | **~10KB** | ~3MB | ~250KB |
| Reflection | **No** | Yes | Yes |
| GraalVM Native Image | **✓** | Partial | Partial |
| Setup | **None** | Annotations | Type tokens |

**Perfect for:**
- Configuration files
- REST API clients
- Microservices (minimal footprint)
- GraalVM native images
- Android apps (avoiding reflection)

## Installation

### Maven

```xml
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>json</artifactId>
    <version>0.1.0</version>
</dependency>
```

## Quick Start

### Parsing

```java
// Parse into generic Object
Object value = Json.parse(json);

// Parse with specific root type
Map<String, Object> object = Json.parseMap(json);
List<Object> array = Json.parseList(json);
String string = Json.parseString(json);
Number number = Json.parseNumber(json);
boolean bool = Json.parseBoolean(json);
```

### Type Mappings

| JSON Type | Java Type |
|-----------|-----------|
| Object | `LinkedHashMap<String, Object>` |
| Array | `ArrayList<Object>` |
| String | `String` |
| Number (integer) | `Long` or `BigInteger` |
| Number (decimal) | `BigDecimal` |
| Boolean | `Boolean` |
| null | `Json.NULL` |

### Serialization

```java
Map<String, Object> data = Map.of(
    "name", "alice",
    "age", 30,
    "active", true
);

// Compact format
String compact = Json.stringify(data);
// {"name":"alice","age":30,"active":true}

// Pretty-printed
String pretty = Json.stringify(data, true);
// {
//   "name": "alice",
//   "age": 30,
//   "active": true
// }
```

### Querying Nested Data

```java
String json = """
    {
        "user": {
            "name": "alice",
            "address": {
                "city": "NYC"
            }
        }
    }
    """;

Map<String, Object> data = Json.parseMap(json);

// Safe navigation with Optional
Optional<String> city = Json.queryString(data, "user", "address", "city");
// Returns Optional.of("NYC")

Optional<Long> age = Json.queryLong(data, "user", "age");
// Returns Optional.empty() (not present)
```

### Configuration Options

```java
// Default options
Json.ParseOptions options = Json.ParseOptions.defaults()
    .decimalsAsBigDecimal(true)    // Use BigDecimal for floats
    .maxDepth(1000)                // Max nesting depth
    .failOnDuplicateKeys(false);   // Reject duplicate keys?

// Parse with custom options
Map<String, Object> data = Json.parseMap(json, options);
```

### Validation

```java
// Check if JSON is valid without parsing
boolean isValid = Json.isValid(json);

// Validate with custom options
boolean valid = Json.isValid(json, Json.ParseOptions.defaults().maxDepth(50));
```

### Error Handling

```java
try {
    Map<String, Object> data = Json.parseMap(json);
} catch (Json.ParseException e) {
    // Get detailed error information
    System.err.println("Error: " + e.getMessage());
    System.err.println("Line: " + e.getLine());
    System.err.println("Column: " + e.getColumn());
    System.err.println("Position: " + e.getPosition());
}
```

## Benchmarks

Run benchmarks locally:

```bash
cd benchmarks
mvn clean package
java -jar target/json-benchmarks.jar
```

Results on typical hardware:

| Operation | Qxotic | Jackson | Difference |
|-----------|--------|---------|------------|
| Parse Small (~500B) | 0.81 µs | 0.88 µs | ~8% faster |
| Round-trip Small | 1.36 µs | 1.70 µs | ~20% faster |

*Benchmarks measured using JMH. See [benchmarks/](benchmarks/) for details.*

# Documentation

- **[Parsing Guide](docs/guides/parsing.md)** - Detailed parsing options and techniques
- **[Serialization Guide](docs/guides/serialization.md)** - Output formatting and custom options
- **[Error Handling](docs/guides/error-handling.md)** - Working with parse exceptions
- **[Migration Guide](docs/guides/migration.md)** - Migrating from Jackson, Gson, or org.json

[badge-license]: https://img.shields.io/badge/license-Apache%202.0-green
[badge-native-image]: https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F
