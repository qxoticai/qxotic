---
sidebar_position: 1
---

# JSON

Strict, minimal JSON parser and printer for Java. RFC 8259 compliant.

Parses JSON directly into standard Java types: `Map` for objects, `List` for arrays, `String`, `Long`/`BigInteger`, `BigDecimal`, `Boolean`, and `Json.NULL`. No custom node objects.

## Installation

import Tabs from '@theme/Tabs';
import TabItem from '@theme/TabItem';

<Tabs>
  <TabItem value="maven" label="Maven">

```xml
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>json</artifactId>
    <version>0.1.0</version>
</dependency>
```

  </TabItem>
  <TabItem value="gradle" label="Gradle">

```groovy
implementation 'com.qxotic:json:0.1.0'
```

  </TabItem>
  <TabItem value="mill" label="Mill">

```scala
mvn"com.qxotic:json:0.1.0"
```

  </TabItem>
</Tabs>

## Quick Start

```snippet path="json/src/test/java/com/qxotic/format/json/snippets/Snippets.java" tag="quick-start"
```

## Parsing

Parse any JSON value:

```snippet path="json/src/test/java/com/qxotic/format/json/snippets/Snippets.java" tag="parse-generic"
```

Use typed methods when you know the root type:

```snippet path="json/src/test/java/com/qxotic/format/json/snippets/Snippets.java" tag="parse-typed-root"
```

Validate without parsing:

```snippet path="json/src/test/java/com/qxotic/format/json/snippets/Snippets.java" tag="parse-validation"
```

## Query Methods

Access nested values without casting:

```snippet path="json/src/test/java/com/qxotic/format/json/snippets/Snippets.java" tag="query-methods"
```

## Serialization

```snippet path="json/src/test/java/com/qxotic/format/json/snippets/Snippets.java" tag="stringify-basic"
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

```snippet path="json/src/test/java/com/qxotic/format/json/snippets/Snippets.java" tag="null-handling"
```

## Parse Options

```snippet path="json/src/test/java/com/qxotic/format/json/snippets/Snippets.java" tag="parse-options-basic"
```

| Option | Default | Description |
|--------|---------|-------------|
| `maxDepth(int)` | 1000 | Maximum nesting depth |
| `decimalsAsBigDecimal(boolean)` | true | `BigDecimal` or `Double` for decimals |
| `failOnDuplicateKeys(boolean)` | false | Reject duplicate object keys |

## Error Handling

`Json.ParseException` includes position, line, column, and a visual caret:

```snippet path="json/src/test/java/com/qxotic/format/json/snippets/Snippets.java" tag="error-handling"
```

Serialization rejects cyclic structures and `NaN`/`Infinity`:

```snippet path="json/src/test/java/com/qxotic/format/json/snippets/Snippets.java" tag="stringify-validation"
```
