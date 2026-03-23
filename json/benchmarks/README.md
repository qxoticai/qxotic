# JSON Benchmarks

JMH benchmarks comparing Qxotic JSON vs Jackson.

## Quick Start

```bash
cd benchmarks
mvn clean package
java -jar target/json-benchmarks.jar
```

## Usage

```bash
# All benchmarks
java -jar target/json-benchmarks.jar

# Specific patterns
java -jar target/json-benchmarks.jar ".*Parse.*"
java -jar target/json-benchmarks.jar ".*Small.*"

# Faster run
java -jar target/json-benchmarks.jar -f 1 -wi 2 -i 3
```

## Results

Test data: Small (~500B), Medium (~50KB), Large (~1MB)

| Operation | Qxotic | Jackson | Difference |
|-----------|--------|---------|------------|
| Parse Small | 0.81 µs/op | 0.88 µs/op | ~8% faster |
| Parse Medium | 90.7 µs/op | 85.0 µs/op | ~7% slower |
| Parse Large | 2739 µs/op | 2255 µs/op | ~21% slower |
| Serialize Small | 0.51 µs/op | 0.46 µs/op | ~11% slower |
| Round-trip Small | 1.36 µs/op | 1.70 µs/op | ~20% faster |
| Throughput Small | 1,188,472 ops/s | 1,101,711 ops/s | ~8% faster |

Qxotic is competitive for small JSON typical in APIs.
