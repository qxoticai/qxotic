# JSON Performance Benchmarks

JMH-based benchmarks comparing Qxotic JSON library vs Jackson (industry standard).

## Quick Start

```bash
# Build and run all benchmarks
cd benchmarks
mvn clean package
java -jar target/json-benchmarks.jar
```

## Benchmark Configuration

The benchmarks compare performance across three JSON sizes:

- **Small**: ~500 bytes (single user object)
- **Medium**: ~50KB (100 user objects)
- **Large**: ~1MB (10 organizations with 100 employees each, nested structure)

## Running Benchmarks

### Run all benchmarks (takes ~2-3 minutes)
```bash
java -jar target/json-benchmarks.jar
```

### Run specific benchmark pattern
```bash
# Only parsing benchmarks
java -jar target/json-benchmarks.jar ".*Parse.*"

# Only small JSON benchmarks
java -jar target/json-benchmarks.jar ".*Small.*"

# Only Qxotic library benchmarks
java -jar target/json-benchmarks.jar ".*qxotic.*"
```

### Adjust measurement parameters
```bash
# Faster run with fewer iterations
java -jar target/json-benchmarks.jar -f 1 -wi 2 -i 3

# More accurate with more iterations
java -jar target/json-benchmarks.jar -f 3 -wi 5 -i 10
```

Parameters:
- `-f 1`: 1 fork (JVM instance)
- `-wi 2`: 2 warmup iterations
- `-i 3`: 3 measurement iterations

## Results Summary

Key comparisons from a sample run:

| Operation | Qxotic | Jackson (Map) | Difference |
|-----------|--------|---------------|------------|
| **Parse Small** | 1.21 µs/op | 0.99 µs/op | ~22% slower |
| **Parse Medium** | 132 µs/op | 96 µs/op | ~37% slower |
| **Parse Large** | 6368 µs/op | 4155 µs/op | ~53% slower |
| **Serialize Small** | 0.72 µs/op | 0.54 µs/op | ~33% slower |
| **Serialize Medium** | 79 µs/op | 54 µs/op | ~46% slower |
| **Serialize Large** | 5830 µs/op | 4154 µs/op | ~40% slower |
| **Round-trip Small** | 1.94 µs/op | 1.68 µs/op | ~15% slower |
| **Round-trip Medium** | 210 µs/op | 140 µs/op | ~50% slower |

## Interpreting Results

- **Average time (avgt)**: Lower is better (microseconds per operation)
- **Throughput (thrpt)**: Higher is better (operations per second)
- **Error bars**: 99.9% confidence intervals

## Test Data

JSON files in `src/main/resources/`:
- `small.json`: Simple user profile
- `medium.json`: Array of 100 users
- `large.json`: Complex nested organization structure

## Performance Notes

Qxotic JSON prioritizes:
- **Simplicity**: Single class, no external dependencies
- **Compliance**: Strict RFC 8259 validation
- **Safety**: Detailed error messages with line/column info

Jackson prioritizes:
- **Performance**: Highly optimized over years
- **Features**: Extensive customization options
- **Ecosystem**: Wide integration with Java frameworks

## Profiling

To identify bottlenecks:

```bash
# Run with async profiler
java -jar target/json-benchmarks.jar -prof async:libPath=/path/to/libasyncProfiler.so

# Run with GC profiler
java -jar target/json-benchmarks.jar -prof gc
```

## Extending Benchmarks

To add new benchmarks, edit `JSONBenchmark.java` and follow the JMH pattern:

```java
@Benchmark
public void myNewBenchmark(Blackhole bh) {
    // Your benchmark code here
    bh.consume(result);
}
```

Then rebuild with `mvn clean package`.
