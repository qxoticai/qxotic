# JSON

Strict JSON parser/printer for Java.

## At a glance

- Parse JSON text into Java values
- Parse typed roots (`object`, `array`, `string`, `number`, `boolean`)
- Serialize compact or pretty JSON
- Validate input quickly (`isValid`)
- Detailed parse errors (`position`, `line`, `column`, caret snippet)

## Installation (Maven)

```xml
<dependency>
    <groupId>com.qxotic</groupId>
    <artifactId>json</artifactId>
    <version>0.1-SNAPSHOT</version>
</dependency>
```

## Quick example

```java
import com.qxotic.format.json.JSON;
import java.util.Map;

Map<String, Object> obj = JSON.parseMap("{\"name\":\"alice\",\"age\":30}");
String compact = JSON.stringify(obj);
String pretty = JSON.stringifyPretty(obj);
```

## API summary

Parse:

- `JSON.parse(CharSequence)`
- `JSON.parse(CharSequence, ParseOptions)`
- `JSON.parseMap(...)`
- `JSON.parseList(...)`
- `JSON.query(...)` / `JSON.queryString(...)` / `JSON.queryMap(...)` / `JSON.queryList(...)` / `JSON.queryNumber(...)` / `JSON.queryBoolean(...)`
- `JSON.isMap(...)` / `JSON.isList(...)` / `JSON.isString(...)` / `JSON.isNumber(...)` / `JSON.isBoolean(...)`
- `JSON.parseString(...)`
- `JSON.parseNumber(...)`
- `JSON.parseBoolean(...)`
- `JSON.isValid(...)`
- `JSON.isNull(...)`

Print:

- `JSON.stringify(value)`
- `JSON.stringify(value, pretty)`
- `JSON.stringifyPretty(value)`

## ParseOptions

- `JSON.options()` (shortcut for defaults)
- `JSON.ParseOptions.strict()`
- `JSON.ParseOptions.fast()`
- `JSON.options().decimalsAsBigDecimal(true)`
- `JSON.options().decimalsAsBigDecimal(false)`
- `maxDepth(int)`
- `decimalsAsBigDecimal(boolean)` / `decimalsAsBigDecimal()`
- `failOnDuplicateKeys(boolean)` / `failOnDuplicateKeys()`

`JSON.parse(...)` and `JSON.isValid(..., options)` throw `NullPointerException` for `null` inputs/options.

## Parsed value model

- object -> `Map<String, Object>` (`LinkedHashMap`)
- array -> `List<Object>`
- string -> `String`
- boolean -> `Boolean`
- null -> `JSON.NULL`
- integers -> `Long` or `BigInteger`
- decimals -> `BigDecimal` (default) or `Double` (with options)

## Behavior notes

- Strict JSON grammar (no trailing commas, no comments)
- Duplicate object keys: last key wins
- Or fail on duplicates with `ParseOptions.failOnDuplicateKeys(true)`
- `stringify` rejects cyclic containers
- `stringify` rejects `NaN` / `Infinity`

## More docs

See `docs/index.md` for full API examples (snippet-backed from compile-checked test code).

## Benchmarks

Performance benchmarks are available in the `benchmarks/` directory, comparing against Jackson.

### Quick Start

```bash
cd benchmarks
mvn clean package
java -jar target/json-benchmarks.jar
```

### Run specific benchmarks

```bash
# Only parsing benchmarks
java -jar target/json-benchmarks.jar ".*Parse.*"

# Only small JSON
java -jar target/json-benchmarks.jar ".*Small.*"

# Faster run with fewer iterations
java -jar target/json-benchmarks.jar -f 1 -wi 2 -i 3
```

See `benchmarks/README.md` for detailed benchmark documentation.
