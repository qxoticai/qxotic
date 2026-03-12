# JSON

[![Java](https://img.shields.io/badge/Java-11+-blue)](https://openjdk.org/projects/jdk/11/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

Strict JSON parser/printer for Java. Zero dependencies, RFC 8259 compliant.

## Features

- **Zero Dependencies**: Single class, no external libraries required
- **RFC 8259 Compliant**: Strict parsing with no deviations from the standard
- **Java-friendly API**: Parse JSON into standard Java classes: `Map`, `List`, `Number`, `Boolean`, `String`...
- **Type Safe**: Typed parsing methods (`parseMap`, `parseList`, etc.) with clear error messages
- **Query API**: Navigate nested structures safely with `queryString()`, `queryNumber()`, etc.
- **Fast Validation**: `isValid()` for quick checks without full parsing overhead
- **Detailed Errors**: Parse exceptions include line, column, and source context
- **Tiny Footprint**: ~10KB vs ~3MB for Jackson/Gson

## Quick Start

### Maven

```xml
<dependency>    
    <groupId>com.qxotic</groupId>
    <artifactId>json</artifactId>
    <version>0.1-SNAPSHOT</version>
</dependency>
```

### Basic Usage

```java
import com.qxotic.format.json.Json;
import java.util.Map;
import java.util.Optional;

// Parse JSON object
Map<String, Object> person = Json.parseMap("{\"name\":\"alice\",\"age\":30}");

// Access fields
String name = (String) person.get("name");

// Navigate nested structures safely
Optional<String> city = Json.queryString(person, "address", "city");

// Serialize back
String compact = Json.stringify(person);
String pretty = Json.stringifyPretty(person);
```

## Features at a Glance

**Parsing:**
- `Json.parse()` - Generic parsing to Object
- `Json.parseMap()` / `Json.parseList()` - Typed root parsing
- `Json.parseString()` / `Json.parseNumber()` / `Json.parseBoolean()` - Scalar parsing
- `Json.query*()` methods - Safe nested navigation with Optional
- `Json.isValid()` - Fast validation without parsing
- `Json.isMap()` / `Json.isList()` / `Json.isNull()` - Type checking

**Serialization:**
- `Json.stringify()` - Compact JSON output
- `Json.stringifyPretty()` - Human-readable with indentation

**Configuration:**
- `Json.options()` - Custom parse options
- `maxDepth()` - Limit nesting depth (security)
- `failOnDuplicateKeys()` - Strict duplicate key handling
- `decimalsAsBigDecimal()` - Precision vs performance trade-off

## Type Mapping

| JSON Type | Java Type | Notes |
|-----------|-----------|-------|
| `object` | `Map<String, Object>` | LinkedHashMap (insertion order) |
| `array` | `List<Object>` | ArrayList |
| `string` | `String` | Full Unicode support |
| `number` (integer) | `Long` or `BigInteger` | Auto-promotes if too large |
| `number` (decimal) | `BigDecimal` | Use options for `Double` |
| `boolean` | `Boolean` | `true` or `false` |
| `null` | `Json.NULL` | Sentinel value (not Java null) |

## Documentation

- [Full Documentation](docs/index.md) - Complete API reference with examples
- [Parsing Guide](docs/guides/parsing.md) - Detailed parsing guide with query methods
- [Serialization Guide](docs/guides/serialization.md) - Serializing Java objects to JSON
- [Error Handling](docs/guides/error-handling.md) - Handling parse exceptions and validation
- [Migration Guide](docs/guides/migration.md) - Moving from Jackson, Gson, or org.json

## Behavior Notes

- **Strict Grammar**: No trailing commas, no comments, strict RFC 8259
- **Duplicate Keys**: Last key wins by default; use `failOnDuplicateKeys(true)` for strict mode
- **Cyclic Detection**: `stringify` rejects cyclic container structures
- **NaN/Infinity**: Rejected during serialization (not valid JSON)
- **Thread Safe**: All `Json` class methods are thread-safe
- **Null Safety**: `Json.parse()` throws `NullPointerException` for null inputs

## Benchmarks

Performance comparison with Jackson (industry standard):

```bash
cd benchmarks
mvn clean package
java -jar target/json-benchmarks.jar
```

**Typical Results:**
- ~20-50% slower than Jackson
- Trade-off: Tiny size vs maximum performance
- Best for: Microservices, Android, CLI tools, libraries

See [benchmarks/README.md](benchmarks/README.md) for detailed results.

## When to Use

**Good fit:**
- Microservices where JAR size matters
- Android apps (APK size reduction)
- GraalVM native images
- Libraries that want minimal dependencies
- Simple JSON processing without POJO mapping

**Not recommended:**
- Heavy POJO mapping requirements
- Complex custom serialization
- Streaming large JSON files
- Need extensive configuration options

