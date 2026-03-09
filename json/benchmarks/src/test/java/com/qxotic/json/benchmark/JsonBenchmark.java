package com.qxotic.json.benchmark;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.qxotic.format.json.Json;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

/**
 * JMH Benchmark comparing Qxotic Json vs Jackson performance.
 * 
 * Run with:
 *   cd benchmarks && mvn clean package && java -jar target/json-benchmarks.jar
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 5, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(1)
@State(Scope.Thread)
public class JsonBenchmark {

    private ObjectMapper jacksonMapper;
    
    // JSON content for different sizes
    private String smallJson;
    private String mediumJson;
    private String largeJson;
    
    // Pre-parsed objects for serialization benchmarks
    private Map<String, Object> qxoticSmallParsed;
    private List<Object> qxoticMediumParsed;
    private Map<String, Object> qxoticLargeParsed;
    
    private JsonNode jacksonSmallParsed;
    private JsonNode jacksonMediumParsed;
    private JsonNode jacksonLargeParsed;
    
    private Map<?, ?> jacksonSmallMap;
    private List<?> jacksonMediumList;
    private Map<?, ?> jacksonLargeMap;

    @Setup
    public void setup() throws IOException {
        jacksonMapper = new ObjectMapper();
        
        // Load generated test JSON files
        smallJson = loadResource("small.json");
        mediumJson = loadResource("medium.json");
        largeJson = loadResource("large.json");
        
        // Pre-parse for serialization benchmarks
        qxoticSmallParsed = Json.parseMap(smallJson);
        qxoticMediumParsed = Json.parseList(mediumJson);
        qxoticLargeParsed = Json.parseMap(largeJson);
        
        jacksonSmallParsed = jacksonMapper.readTree(smallJson);
        jacksonMediumParsed = jacksonMapper.readTree(mediumJson);
        jacksonLargeParsed = jacksonMapper.readTree(largeJson);
        
        jacksonSmallMap = jacksonMapper.readValue(smallJson, Map.class);
        jacksonMediumList = jacksonMapper.readValue(mediumJson, List.class);
        jacksonLargeMap = jacksonMapper.readValue(largeJson, Map.class);
    }
    
    private String loadResource(String name) throws IOException {
        try (InputStream is = getClass().getClassLoader().getResourceAsStream(name)) {
            if (is == null) {
                throw new IOException("Resource not found: " + name);
            }
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    // ==================== PARSING BENCHMARKS ====================
    
    @Benchmark
    public void qxoticParseSmall(Blackhole bh) {
        bh.consume(Json.parseMap(smallJson));
    }
    
    @Benchmark
    public void jacksonParseSmallTree(Blackhole bh) throws IOException {
        bh.consume(jacksonMapper.readTree(smallJson));
    }
    
    @Benchmark
    public void jacksonParseSmallMap(Blackhole bh) throws IOException {
        bh.consume(jacksonMapper.readValue(smallJson, Map.class));
    }
    
    @Benchmark
    public void qxoticParseMedium(Blackhole bh) {
        bh.consume(Json.parseList(mediumJson));
    }
    
    @Benchmark
    public void jacksonParseMediumTree(Blackhole bh) throws IOException {
        bh.consume(jacksonMapper.readTree(mediumJson));
    }
    
    @Benchmark
    public void jacksonParseMediumList(Blackhole bh) throws IOException {
        bh.consume(jacksonMapper.readValue(mediumJson, List.class));
    }
    
    @Benchmark
    public void qxoticParseLarge(Blackhole bh) {
        bh.consume(Json.parseMap(largeJson));
    }
    
    @Benchmark
    public void jacksonParseLargeTree(Blackhole bh) throws IOException {
        bh.consume(jacksonMapper.readTree(largeJson));
    }
    
    @Benchmark
    public void jacksonParseLargeMap(Blackhole bh) throws IOException {
        bh.consume(jacksonMapper.readValue(largeJson, Map.class));
    }

    // ==================== SERIALIZATION BENCHMARKS ====================
    
    @Benchmark
    public void qxoticSerializeSmall(Blackhole bh) {
        bh.consume(Json.stringify(qxoticSmallParsed));
    }
    
    @Benchmark
    public void jacksonSerializeSmallFromTree(Blackhole bh) throws IOException {
        bh.consume(jacksonMapper.writeValueAsString(jacksonSmallParsed));
    }
    
    @Benchmark
    public void jacksonSerializeSmallFromMap(Blackhole bh) throws IOException {
        bh.consume(jacksonMapper.writeValueAsString(jacksonSmallMap));
    }
    
    @Benchmark
    public void qxoticSerializeMedium(Blackhole bh) {
        bh.consume(Json.stringify(qxoticMediumParsed));
    }
    
    @Benchmark
    public void jacksonSerializeMediumFromTree(Blackhole bh) throws IOException {
        bh.consume(jacksonMapper.writeValueAsString(jacksonMediumParsed));
    }
    
    @Benchmark
    public void jacksonSerializeMediumFromList(Blackhole bh) throws IOException {
        bh.consume(jacksonMapper.writeValueAsString(jacksonMediumList));
    }
    
    @Benchmark
    public void qxoticSerializeLarge(Blackhole bh) {
        bh.consume(Json.stringify(qxoticLargeParsed));
    }
    
    @Benchmark
    public void jacksonSerializeLargeFromTree(Blackhole bh) throws IOException {
        bh.consume(jacksonMapper.writeValueAsString(jacksonLargeParsed));
    }
    
    @Benchmark
    public void jacksonSerializeLargeFromMap(Blackhole bh) throws IOException {
        bh.consume(jacksonMapper.writeValueAsString(jacksonLargeMap));
    }

    // ==================== ROUND-TRIP BENCHMARKS ====================
    
    @Benchmark
    public void qxoticRoundTripSmall(Blackhole bh) {
        String serialized = Json.stringify(qxoticSmallParsed);
        bh.consume(Json.parseMap(serialized));
    }
    
    @Benchmark
    public void jacksonRoundTripSmall(Blackhole bh) throws IOException {
        String serialized = jacksonMapper.writeValueAsString(jacksonSmallMap);
        bh.consume(jacksonMapper.readValue(serialized, Map.class));
    }
    
    @Benchmark
    public void qxoticRoundTripMedium(Blackhole bh) {
        String serialized = Json.stringify(qxoticMediumParsed);
        bh.consume(Json.parseList(serialized));
    }
    
    @Benchmark
    public void jacksonRoundTripMedium(Blackhole bh) throws IOException {
        String serialized = jacksonMapper.writeValueAsString(jacksonMediumList);
        bh.consume(jacksonMapper.readValue(serialized, List.class));
    }

    // ==================== THROUGHPUT BENCHMARKS ====================
    
    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @OutputTimeUnit(TimeUnit.SECONDS)
    public void qxoticParseSmallThroughput(Blackhole bh) {
        bh.consume(Json.parseMap(smallJson));
    }
    
    @Benchmark
    @BenchmarkMode(Mode.Throughput)
    @OutputTimeUnit(TimeUnit.SECONDS)
    public void jacksonParseSmallThroughput(Blackhole bh) throws IOException {
        bh.consume(jacksonMapper.readValue(smallJson, Map.class));
    }
}
