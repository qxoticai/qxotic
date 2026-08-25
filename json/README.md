# json

[![Maven Central](https://img.shields.io/maven-central/v/com.qxotic/json)](https://search.maven.org/artifact/com.qxotic/json)
[![Java](https://img.shields.io/badge/Java-11+-blue)](https://openjdk.org/projects/jdk/11/)
[![License][badge-license]](LICENSE)
[![GraalVM Native Image][badge-native-image]](https://www.graalvm.org/latest/reference-manual/native-image/)

**JSON for the JVM, minus the baggage.** A strict, minimal RFC 8259 parser and printer that maps
JSON straight onto standard Java collections. **~10 KB. Zero dependencies. Zero reflection.**

```java
Map<String, Object> data = Json.parseMap("{\"name\":\"alice\",\"age\":30}");
String json = Json.stringify(data);
```

## Why not Jackson?

| | **qxotic json** | Jackson | Gson |
|---|---|---|---|
| Dependencies | **0** | 3+ | 1+ |
| JAR size | **~10 KB** | ~3 MB | ~250 KB |
| Reflection | **None** | Yes | Yes |
| GraalVM native image | **Out of the box** | Partial | Partial |
| Setup | **None** | Annotations, modules | Type tokens |

Perfect for config files, REST clients, microservices, Android, and native images — anywhere a
3 MB reflection-driven mapper is overkill.

## Quick start

```xml
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>json</artifactId>
    <version>0.2.0</version>
</dependency>
```

### Parse

```java
Object any    = Json.parse(json);          // generic
Map<String, Object> obj = Json.parseMap(json);
List<Object> arr        = Json.parseList(json);
```

| JSON type | Java type |
|-----------|-----------|
| object | `LinkedHashMap<String, Object>` |
| array | `ArrayList<Object>` |
| number (integer) | `Long` or `BigInteger` |
| number (decimal) | `BigDecimal` |
| null | `Json.NULL` |

### Print

```java
Json.stringify(data);        // {"name":"alice","age":30}
Json.stringify(data, true);  // pretty-printed
```

### Query safely

```java
Optional<String> city = Json.queryString(data, "user", "address", "city");
```

### Errors that point

```java
catch (Json.ParseException e) {
    e.getLine(); e.getColumn(); e.getPosition();   // exact location, every time
}
```

Options (depth limits, duplicate-key strictness, decimal handling) and validation without parsing
(`Json.isValid`) are covered in the [documentation](https://qxotic.ai/docs/json).

## Benchmarks

JMH, typical hardware:

| Operation | qxotic | Jackson | Difference |
|-----------|--------|---------|------------|
| Parse small (~500 B) | 0.81 µs | 0.88 µs | ~8% faster |
| Round-trip small | 1.36 µs | 1.70 µs | ~20% faster |

Reproduce: `cd benchmarks && mvn package && java -jar target/json-benchmarks.jar`.

## Documentation

**[qxotic.ai/docs/json](https://qxotic.ai/docs/json)** — parsing, serialization, error handling,
and migration from Jackson/Gson.

## License

Apache 2.0

[badge-license]: https://img.shields.io/badge/license-Apache%202.0-green
[badge-native-image]: https://img.shields.io/badge/GraalVM-Native_Image-F29111?labelColor=00758F
