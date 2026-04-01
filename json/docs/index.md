# JSON

Strict, minimal JSON parser and printer for Java. RFC 8259 compliant, zero dependencies, single class.

Parses JSON directly into standard Java types: `Map` for objects, `List` for arrays, `String`, `Long`/`BigInteger`, `BigDecimal`, `Boolean`, and `Json.NULL`. No custom node objects, just the collections and values you already use.

## Installation

=== "Maven"

    ```xml
    <dependency>
        <groupId>com.qxotic</groupId>
        <artifactId>json</artifactId>
        <version>0.1.0</version>
    </dependency>
    ```

=== "Gradle"

    ```groovy
    implementation 'com.qxotic:json:0.1.0'
    ```

=== "Mill"

    ```scala
    mvn"com.qxotic:json:0.1.0"
    ```

## Quick Start

```java
--8<-- "Snippets.java:quick-start"
```

## Parsing

Parse any JSON value:

```java
--8<-- "Snippets.java:parse-generic"
```

Use typed methods when you know the root type:

```java
--8<-- "Snippets.java:parse-typed-root"
```

Validate without parsing:

```java
--8<-- "Snippets.java:parse-validation"
```

## Query Methods

Access nested values without casting:

```java
--8<-- "Snippets.java:query-methods"
```

## Serialization

```java
--8<-- "Snippets.java:stringify-basic"
```

## Type Mapping

| JSON | Java | Notes |
|------|------|-------|
| `object` | `Map<String, Object>` | LinkedHashMap, insertion order preserved |
| `array` | `List<Object>` | ArrayList |
| `string` | `String` | |
| `number` (integer) | `Long` / `BigInteger` | Auto-promotes on overflow |
| `number` (decimal) | `BigDecimal` | `Double` with `decimalsAsBigDecimal(false)` |
| `boolean` | `Boolean` | |
| `null` | `Json.NULL` | Sentinel, not Java `null` |

## Null Handling

```java
--8<-- "Snippets.java:null-handling"
```

## Parse Options

```java
--8<-- "Snippets.java:parse-options-basic"
```

| Option | Default | Description |
|--------|---------|-------------|
| `maxDepth(int)` | 1000 | Maximum nesting depth |
| `decimalsAsBigDecimal(boolean)` | true | `BigDecimal` or `Double` for decimals |
| `failOnDuplicateKeys(boolean)` | false | Reject duplicate object keys |

## Error Handling

`Json.ParseException` includes position, line, column, and a visual caret:

```java
--8<-- "Snippets.java:error-handling"
```

Serialization rejects cyclic structures and `NaN`/`Infinity`:

```java
--8<-- "Snippets.java:stringify-validation"
```
