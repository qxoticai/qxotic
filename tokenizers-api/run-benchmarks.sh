#!/bin/bash
#
# Run JMH benchmarks and calculate chars/s and tokens/s metrics
#

set -e

echo "Running Tokenizer Benchmarks..."
echo "================================"
echo ""

# Run benchmarks with JSON output for parsing
mvn exec:java \
    -Dexec.mainClass="org.openjdk.jmh.Main" \
    -Dexec.classpathScope=test \
    -Dexec.args="ai.qxotic.tokenizers.benchmarks.TokenizerBenchmark -f 0 -wi 3 -i 5 -rf json -rff target/benchmark-results.json" \
    2>&1 | tee target/benchmark-output.txt

echo ""
echo "================================"
echo "Benchmark Complete!"
echo ""
echo "Results saved to: target/benchmark-results.json"
echo ""
echo "To calculate chars/s and tokens/s from the results:"
echo ""
echo "Formula:"
echo "  chars/s  = ops/s × text_length"
echo "  tokens/s = ops/s × token_count"
echo ""
echo "Example calculation for 'medium' text (~1000 chars):"
echo "  If ops/s = 33,550"
echo "  Then chars/s = 33,550 × 1000 = 33,550,000 chars/s"
echo ""
echo "Text sizes:"
echo "  small  = ~100 chars"
echo "  medium = ~1000 chars"
echo "  large  = ~10000 chars"
