# JSON

Strict, minimal JSON parser and printer for Java. RFC 8259 compliant, zero dependencies.

## Features

- **RFC 8259 compliant**: Strict parsing, no deviations
- **Type-safe APIs**: `parseMap()`, `parseList()`, typed root parsing
- **Precise decimals**: `BigDecimal` by default, optional `Double`
- **Detailed errors**: Line/column positions with source context
- **Zero dependencies**: Single class, no external requirements

## What is JSON?

JSON (JavaScript Object Notation) is a lightweight data interchange format. This library provides:

- **Parsing**: Convert JSON strings to Java objects
- **Serialization**: Convert Java objects to JSON strings
- **Validation**: Check JSON validity without parsing
- **Type mapping**: JSON values map to specific Java types

!!! info "Library Scope"
    This library provides **parsing and serialization only**. It does **not**:
    - Schema validation (beyond JSON syntax)
    - Streaming/chunked parsing
    - Custom type adapters

## Quick Start

```java
import com.qxotic.format.json.JSON;
import java.util.Map;
import java.util.List;

// Parse JSON string
Map<String, Object> config = JSON.parseMap("{"host":"localhost","port":8080}");

String host = (String) config.get("host");
Long port = (Long) config.get("port");

// Serialize back to JSON
String json = JSON.stringify(config);
```

## Java Type Mappings

JSON values are mapped to Java types as follows:

### Objects

JSON objects become `Map<String, Object>`:

```java
// {"name": "alice", "age": 30}
Map<String, Object> person = JSON.parseMap(json);
String name = (String) person.get("name");
Long age = (Long) person.get("age");
```

The map implementation preserves insertion order (LinkedHashMap).

### Arrays

JSON arrays become `List<Object>`:

```java
// [1, 2, 3]
List<Object> numbers = JSON.parseList(json);
Long first = (Long) numbers.get(0);
```

### Strings

JSON strings become `String`:

```java
// "hello world"
String text = JSON.parseString(json);
```

### Numbers

JSON numbers become different Java types based on value:

| JSON Value | Java Type | Example |
|------------|-----------|---------|
| Integer within `long` range | `Long` | `42` → `Long` |
| Integer exceeds `long` range | `BigInteger` | `99999999999999999999` → `BigInteger` |
| Decimal (default) | `BigDecimal` | `3.14159` → `BigDecimal` |
| Decimal (optional) | `Double` | `3.14159` → `Double` |

```java
// Integer - fits in long
Number small = JSON.parseNumber("42");
// small instanceof Long

// Integer - exceeds long range
Number big = JSON.parseNumber("99999999999999999999");
// big instanceof BigInteger

// Decimal - precise (default)
Number precise = JSON.parseNumber("0.1");
// precise instanceof BigDecimal

// Decimal - fast approximation
JSON.ParseOptions fast = JSON.options().decimalsAsBigDecimal(false);
Number approx = JSON.parseNumber("0.1", fast);
// approx instanceof Double
```

!!! tip "Decimal Precision"
    Use `BigDecimal` (default) for financial or scientific data requiring exact precision. Use `Double` for better performance when approximate precision is acceptable.

### Booleans

JSON booleans become `Boolean`:

```java
// true or false
Boolean active = JSON.parseBoolean("true");
```

### Null

JSON `null` becomes a special sentinel value `JSON.NULL` (not Java `null`):

```java
Map<String, Object> obj = JSON.parseMap("{"value":null}");
Object value = obj.get("value");

// Check for JSON null
if (JSON.isNull(value)) {
    // Key exists, value is JSON null
}

// Distinguish from missing key
Object missing = obj.get("missing");
if (missing == null) {
    // Key doesn't exist
}
```

## Type Mapping Summary

| JSON Type | Default Java Type | Notes |
|-----------|-------------------|-------|
| `object` | `Map<String, Object>` | Insertion order preserved |
| `array` | `List<Object>` | ArrayList implementation |
| `string` | `String` | Unicode fully supported |
| `number` (integer) | `Long` or `BigInteger` | Auto-promotes if too large |
| `number` (decimal) | `BigDecimal` | Precise; use options for `Double` |
| `boolean` | `Boolean` | `true` or `false` |
| `null` | `JSON.NULL` | Sentinel value, not Java `null` |

## Installation

=== "Maven"

    ```xml
    <dependency>
        <groupId>com.qxotic</groupId>
        <artifactId>json</artifactId>
        <version>0.1-SNAPSHOT</version>
    </dependency>
    ```

=== "Gradle"

    ```groovy
    implementation 'com.qxotic:json:0.1-SNAPSHOT'
    ```

=== "Mill"

    ```scala
    ivy"com.qxotic::json:0.1-SNAPSHOT"
    ```

## Parsing JSON

### Generic Parsing

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:parse-generic"
```

### Typed Root Parsing

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:parse-typed-root"
```

### Query Methods

Navigate nested structures safely with type-safe query methods:

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:query-methods"
```

### Object Access Patterns

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:parse-object-access"
```

### Validation

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:parse-validation"
```

## Parse Options

### Basic Options

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:parse-options-basic"
```

### Decimal Handling

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:parse-options-decimals"
```

### Depth Limiting

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:parse-options-depth"
```

### Duplicate Key Handling

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:parse-options-duplicate"
```

## Serializing to JSON

### Basic Serialization

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:stringify-basic"
```

### Pretty Printing

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:stringify-pretty"
```

### Round-Trip Example

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:round-trip"
```

## Data Model Details

### Working with Numbers

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:data-model"
```

### Null Handling

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:null-handling"
```

## Error Handling

### Parse Exceptions

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:error-handling"
```

### Common Error Types

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:error-types"
```

## Validation Constraints

### Cyclic Structure Detection

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:stringify-validation"
```

## Unicode Support

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:unicode"
```

## GGUF Integration Example

```java
--8<-- "json/src/test/java/com/qxotic/format/json/Snippets.java:gguf-integration"
```

## Thread Safety

!!! info "Concurrency"
    - `JSON` class methods are **thread-safe** and can be called concurrently
    - `ParseOptions` instances are **immutable** and reusable across threads
    - Parsed objects (Map, List) are standard Java collections - not thread-safe for concurrent modification

## Next Steps

- [Parsing Guide](guides/parsing.md) - Parse options, typed roots, validation
- [Serialization Guide](guides/serialization.md) - Compact vs pretty, round-trips
- [Error Handling](guides/error-handling.md) - Exception details, common errors
