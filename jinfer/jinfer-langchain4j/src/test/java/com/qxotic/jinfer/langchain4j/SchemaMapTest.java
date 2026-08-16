package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.ToolCallSyntax;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.internal.JsonSchemaElementUtils;
import dev.langchain4j.model.chat.request.json.JsonAnyOfSchema;
import dev.langchain4j.model.chat.request.json.JsonArraySchema;
import dev.langchain4j.model.chat.request.json.JsonBooleanSchema;
import dev.langchain4j.model.chat.request.json.JsonEnumSchema;
import dev.langchain4j.model.chat.request.json.JsonIntegerSchema;
import dev.langchain4j.model.chat.request.json.JsonNullSchema;
import dev.langchain4j.model.chat.request.json.JsonNumberSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonRawSchema;
import dev.langchain4j.model.chat.request.json.JsonReferenceSchema;
import dev.langchain4j.model.chat.request.json.JsonSchemaElement;
import dev.langchain4j.model.chat.request.json.JsonStringSchema;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * {@link Mappings#toSchemaMap} against langchain4j's own {@code JsonSchemaElementUtils.toMap} as
 * the oracle, over every element type. Our converter exists so PRODUCTION code touches only the
 * public {@code dev.langchain4j.model.chat.request.json} types - an internal class may change in a
 * patch release, and this conversion feeds both the tool declarations models are trained on and the
 * grammar that constrains structured output.
 *
 * <p>The oracle import is the deliberate exception: as a TEST it turns an upstream semantic change
 * into a red build here, which is exactly the warning we want, instead of a silently different
 * prompt in production. If langchain4j ever removes the class, delete this one test - the converter
 * and its behaviour are unaffected.
 */
class SchemaMapTest {

    static Stream<Object[]> schemas() {
        return Stream.of(
                new Object[] {"string", new JsonStringSchema()},
                new Object[] {
                    "string with description",
                    JsonStringSchema.builder().description("the city name").build()
                },
                new Object[] {"integer", new JsonIntegerSchema()},
                new Object[] {"number", new JsonNumberSchema()},
                new Object[] {"boolean", new JsonBooleanSchema()},
                new Object[] {"null", new JsonNullSchema()},
                new Object[] {
                    "enum",
                    JsonEnumSchema.builder()
                            .enumValues("CELSIUS", "FAHRENHEIT")
                            .description("unit")
                            .build()
                },
                new Object[] {
                    "array of strings",
                    JsonArraySchema.builder()
                            .items(new JsonStringSchema())
                            .description("tags")
                            .build()
                },
                new Object[] {
                    "flat object",
                    JsonObjectSchema.builder()
                            .addStringProperty("city", "the city")
                            .addIntegerProperty("days")
                            .required("city")
                            .build()
                },
                new Object[] {
                    "object without required",
                    JsonObjectSchema.builder().addStringProperty("note").build()
                },
                new Object[] {
                    "object with additionalProperties",
                    JsonObjectSchema.builder()
                            .addStringProperty("city")
                            .additionalProperties(false)
                            .build()
                },
                new Object[] {
                    "nested object and array",
                    JsonObjectSchema.builder()
                            .addProperty(
                                    "location",
                                    JsonObjectSchema.builder()
                                            .addNumberProperty("lat")
                                            .addNumberProperty("lon")
                                            .required("lat", "lon")
                                            .build())
                            .addProperty(
                                    "waypoints",
                                    JsonArraySchema.builder()
                                            .items(
                                                    JsonObjectSchema.builder()
                                                            .addStringProperty("name")
                                                            .build())
                                            .build())
                            .required("location")
                            .build()
                },
                new Object[] {
                    "anyOf",
                    JsonAnyOfSchema.builder()
                            .description("either")
                            .anyOf(new JsonStringSchema(), new JsonIntegerSchema())
                            .build()
                },
                new Object[] {
                    "reference with definitions",
                    JsonObjectSchema.builder()
                            .addProperty(
                                    "self", JsonReferenceSchema.builder().reference("Node").build())
                            .definitions(
                                    Map.of(
                                            "Node",
                                            JsonObjectSchema.builder()
                                                    .addStringProperty("id")
                                                    .build()))
                            .build()
                },
                new Object[] {
                    "top-level reference", JsonReferenceSchema.builder().reference("Root").build()
                },
                new Object[] {"empty object", JsonObjectSchema.builder().build()},
                new Object[] {
                    "object with only a description",
                    JsonObjectSchema.builder().description("opaque").build()
                },
                new Object[] {
                    "array of arrays",
                    JsonArraySchema.builder()
                            .items(JsonArraySchema.builder().items(new JsonIntegerSchema()).build())
                            .build()
                },
                new Object[] {
                    "array of enums",
                    JsonArraySchema.builder()
                            .items(JsonEnumSchema.builder().enumValues("A", "B").build())
                            .build()
                },
                new Object[] {
                    "three levels deep",
                    JsonObjectSchema.builder()
                            .addProperty(
                                    "a",
                                    JsonObjectSchema.builder()
                                            .addProperty(
                                                    "b",
                                                    JsonObjectSchema.builder()
                                                            .addStringProperty("c")
                                                            .required("c")
                                                            .build())
                                            .required("b")
                                            .build())
                            .required("a")
                            .build()
                },
                new Object[] {
                    "scalars with descriptions",
                    JsonObjectSchema.builder()
                            .addProperty(
                                    "i", JsonIntegerSchema.builder().description("count").build())
                            .addProperty(
                                    "n", JsonNumberSchema.builder().description("ratio").build())
                            .addProperty(
                                    "b", JsonBooleanSchema.builder().description("flag").build())
                            .build()
                },
                new Object[] {
                    "several definitions",
                    JsonObjectSchema.builder()
                            .addProperty(
                                    "left", JsonReferenceSchema.builder().reference("Node").build())
                            .addProperty(
                                    "right",
                                    JsonReferenceSchema.builder().reference("Leaf").build())
                            .definitions(
                                    new LinkedHashMap<>(
                                            Map.of(
                                                    "Node",
                                                    JsonObjectSchema.builder()
                                                            .addStringProperty("id")
                                                            .build(),
                                                    "Leaf",
                                                    JsonObjectSchema.builder()
                                                            .addIntegerProperty("value")
                                                            .build())))
                            .build()
                },
                new Object[] {
                    "anyOf of object and reference",
                    JsonAnyOfSchema.builder()
                            .anyOf(
                                    JsonObjectSchema.builder().addStringProperty("kind").build(),
                                    JsonReferenceSchema.builder().reference("Other").build())
                            .build()
                },
                new Object[] {
                    "raw with nesting, unicode and escapes",
                    JsonRawSchema.from(
                            "{\"type\":\"object\",\"description\":\"caf\\u00e9 \\\"quoted\\\"\","
                                    + "\"properties\":{\"xs\":{\"type\":\"array\",\"items\":"
                                    + "{\"type\":\"number\"}}},\"required\":[\"xs\"]}")
                },
                new Object[] {
                    "raw",
                    JsonRawSchema.from(
                            "{\"type\":\"object\",\"properties\":{\"n\":{\"type\":\"integer\"}}}")
                });
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("schemas")
    void matchesLangchain4jsOwnConversion(String name, JsonSchemaElement element) {
        Map<String, Object> theirs = JsonSchemaElementUtils.toMap(element);
        Map<String, Object> mine = Mappings.toSchemaMap(element);
        assertEquals(theirs, mine, name);
        // and byte-for-byte once rendered: these maps become PROMPT TEXT through the same
        // canonical writer the templates use, so key order is part of the contract, not a detail
        assertEquals(
                ToolCallSyntax.jinjaJson(theirs),
                ToolCallSyntax.jinjaJson(mine),
                name + " (rendered)");
    }

    @Test
    void rawSchemaKeepsItsJsonTypes() {
        // the raw branch parses rather than re-describes: numbers stay numbers, booleans booleans,
        // null null - a stringified value here would silently change the grammar it compiles to
        Map<String, Object> map =
                Mappings.toSchemaMap(
                        JsonRawSchema.from(
                                "{\"type\":\"object\",\"minItems\":2,\"strict\":true,"
                                        + "\"default\":null,\"ratio\":0.5}"));
        assertEquals("object", map.get("type"));
        assertEquals(2, ((Number) map.get("minItems")).intValue());
        assertEquals(Boolean.TRUE, map.get("strict"));
        assertEquals(0.5, ((Number) map.get("ratio")).doubleValue());
        assertTrue(map.containsKey("default") && map.get("default") == null, "null stays null");
    }

    @Test
    void rawSchemaThatIsNotAnObjectIsRefused() {
        UnsupportedFeatureException e =
                assertThrows(
                        UnsupportedFeatureException.class,
                        () -> Mappings.toSchemaMap(JsonRawSchema.from("[1, 2, 3]")));
        assertTrue(e.getMessage().contains("JSON object"), e.getMessage());
    }

    @Test
    void unknownElementTypeIsRefusedByName() {
        // a langchain4j upgrade adding a schema element must fail loudly here, never render a
        // silently empty schema into a prompt or a grammar
        JsonSchemaElement exotic = () -> "from the future";
        UnsupportedFeatureException e =
                assertThrows(UnsupportedFeatureException.class, () -> Mappings.toSchemaMap(exotic));
        assertTrue(e.getMessage().contains("unsupported JSON schema element"), e.getMessage());
    }

    @Test
    void richToolSchemaRendersTheCanonicalDeclaration() {
        // the whole point of the conversion: what a model actually reads. Nested object, array of
        // enums, optional property - all in the Jinja-tojson canonical form the templates weld in
        ToolSpecification spec =
                ToolSpecification.builder()
                        .name("book_flight")
                        .description("Book a flight")
                        .parameters(
                                JsonObjectSchema.builder()
                                        .addStringProperty("destination", "IATA code")
                                        .addProperty(
                                                "when",
                                                JsonObjectSchema.builder()
                                                        .addStringProperty("date")
                                                        .required("date")
                                                        .build())
                                        .addProperty(
                                                "classes",
                                                JsonArraySchema.builder()
                                                        .items(
                                                                JsonEnumSchema.builder()
                                                                        .enumValues(
                                                                                "ECONOMY",
                                                                                "BUSINESS")
                                                                        .build())
                                                        .build())
                                        .required("destination", "when")
                                        .build())
                        .build();
        // compact JSON, insertion order preserved - the canonical tojson rendering (with
        // ", "/": " separators) is the Jinja layer's job, tested in jinfer-chat
        assertEquals(
                "{\"type\":\"function\",\"function\":{\"name\":\"book_flight\","
                        + "\"description\":\"Book a flight\",\"parameters\":{\"type\":"
                        + "\"object\",\"properties\":{\"destination\":{\"type\":\"string\","
                        + "\"description\":\"IATA code\"},\"when\":{\"type\":\"object\","
                        + "\"properties\":{\"date\":{\"type\":\"string\"}},\"required\":"
                        + "[\"date\"]},\"classes\":{\"type\":\"array\",\"items\":{\"type\":"
                        + "\"string\",\"enum\":[\"ECONOMY\",\"BUSINESS\"]}}},\"required\":"
                        + "[\"destination\",\"when\"]}}}",
                com.qxotic.format.json.Json.stringify(
                        Mappings.toTools(List.of(spec)).get(0).definition()));
    }

    @Test
    void objectPropertiesKeepDeclarationOrder() {
        // the tool declarations models were trained on are ORDERED text: a reordered properties map
        // renders a different prompt, so the conversion must not sort or hash the keys
        Map<String, Object> map =
                Mappings.toSchemaMap(
                        JsonObjectSchema.builder()
                                .addStringProperty("zebra")
                                .addStringProperty("apple")
                                .addStringProperty("mango")
                                .build());
        @SuppressWarnings("unchecked")
        Map<String, Object> properties = (Map<String, Object>) map.get("properties");
        assertEquals(List.of("zebra", "apple", "mango"), List.copyOf(properties.keySet()));
    }
}
