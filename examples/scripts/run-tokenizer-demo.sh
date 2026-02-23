#!/bin/bash
# Run the Tokenizer Demo Server

cd "$(dirname "$0")/.."

echo "Building tokenizer demo..."
mvn compile -pl examples -am -DskipTests -q

if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

echo "Starting tokenizer demo server..."
echo "Open http://localhost:8080 in your browser"
echo ""

mvn exec:java -pl examples -Dexec.mainClass="com.qxotic.jota.examples.tokenizer.TokenizerDemoServer" -q
