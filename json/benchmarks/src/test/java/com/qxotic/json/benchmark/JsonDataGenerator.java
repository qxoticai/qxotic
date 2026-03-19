package com.qxotic.json.benchmark;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * Generates JSON test files for benchmarks. Run during build to generate files dynamically instead
 * of storing in repo.
 */
public class JsonDataGenerator {

    private static final Random RANDOM = new Random(42); // Fixed seed for reproducibility

    public static void main(String[] args) throws IOException {
        Path outputDir = Paths.get(args.length > 0 ? args[0] : "target/test-classes");
        Files.createDirectories(outputDir);

        System.out.println("Generating JSON benchmark data...");

        // Generate small.json - single user object
        generateSmallJson(outputDir.resolve("small.json"));
        System.out.println("Generated small.json");

        // Generate medium.json - array of 100 users
        generateMediumJson(outputDir.resolve("medium.json"));
        System.out.println("Generated medium.json");

        // Generate large.json - complex nested structure
        generateLargeJson(outputDir.resolve("large.json"));
        System.out.println("Generated large.json");

        System.out.println("Done!");
    }

    private static void generateSmallJson(Path path) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"id\": \"user-12345\",\n");
        sb.append("  \"name\": \"Alice Johnson\",\n");
        sb.append("  \"email\": \"alice@example.com\",\n");
        sb.append("  \"active\": true,\n");
        sb.append("  \"age\": 28,\n");
        sb.append("  \"balance\": 1234.56,\n");
        sb.append("  \"tags\": [\"developer\", \"java\", \"json\"],\n");
        sb.append("  \"address\": {\n");
        sb.append("    \"street\": \"123 Main St\",\n");
        sb.append("    \"city\": \"San Francisco\",\n");
        sb.append("    \"zip\": \"94102\"\n");
        sb.append("  },\n");
        sb.append("  \"metadata\": {\n");
        sb.append("    \"created\": \"2024-01-15T10:30:00Z\",\n");
        sb.append("    \"updated\": null\n");
        sb.append("  }\n");
        sb.append("}\n");
        Files.writeString(path, sb.toString());
    }

    private static void generateMediumJson(Path path) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("[\n");

        String[] cities = {"Chicago", "Austin", "Seattle", "Boston", "Denver"};

        for (int i = 0; i < 100; i++) {
            if (i > 0) sb.append(",\n");
            String city = cities[i % cities.length];
            boolean active = i % 3 != 0;

            sb.append("  {\n");
            sb.append(String.format("    \"id\": \"user-%05d\",\n", i));
            sb.append(String.format("    \"name\": \"User %d\",\n", i));
            sb.append(String.format("    \"email\": \"user%d@example.com\",\n", i));
            sb.append(String.format("    \"active\": %b,\n", active));
            sb.append(String.format("    \"age\": %d,\n", 20 + (i % 50)));
            sb.append(
                    String.format("    \"balance\": %.2f,\n", 1000.0 + RANDOM.nextDouble() * 9000));
            sb.append(
                    String.format(
                            "    \"tags\": [\"user\", \"group-%d\", \"%s\"],\n",
                            i % 10, active ? "active" : "inactive"));
            sb.append("    \"address\": {\n");
            sb.append(
                    String.format(
                            "      \"street\": \"%d Street Name\",\n", RANDOM.nextInt(10000)));
            sb.append(String.format("      \"city\": \"%s\",\n", city));
            sb.append(String.format("      \"zip\": \"%05d\"\n", 10000 + RANDOM.nextInt(90000)));
            sb.append("    },\n");
            sb.append("    \"metadata\": {\n");
            sb.append(
                    String.format(
                            "      \"created\": \"2024-%02d-%02dT10:30:00Z\",\n",
                            1 + (i % 12), 1 + (i % 28)));
            sb.append("      \"updated\": null,\n");
            sb.append(String.format("      \"login_count\": %d\n", RANDOM.nextInt(1000)));
            sb.append("    }\n");
            sb.append("  }");
        }

        sb.append("\n]\n");
        Files.writeString(path, sb.toString());
    }

    private static void generateLargeJson(Path path) throws IOException {
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"version\": \"2.0\",\n");
        sb.append("  \"timestamp\": \"2024-01-15T10:30:00Z\",\n");
        sb.append("  \"metadata\": {\n");
        sb.append("    \"total_records\": 1000,\n");
        sb.append("    \"generated_by\": \"benchmark-generator\",\n");
        sb.append("    \"compression\": false\n");
        sb.append("  },\n");
        sb.append("  \"organizations\": [\n");

        for (int org = 0; org < 10; org++) {
            if (org > 0) sb.append(",\n");
            generateOrganization(sb, org);
        }

        sb.append("\n  ]\n");
        sb.append("}\n");
        Files.writeString(path, sb.toString());
    }

    private static void generateOrganization(StringBuilder sb, int orgId) {
        sb.append("    {\n");
        sb.append(String.format("      \"id\": \"org-%03d\",\n", orgId));
        sb.append(String.format("      \"name\": \"Organization %d\",\n", orgId));
        sb.append(String.format("      \"active\": %b,\n", orgId % 3 != 0));
        sb.append(String.format("      \"revenue\": %.2f,\n", RANDOM.nextDouble() * 100000000));
        sb.append("      \"employees\": [\n");

        // Generate 100 employees per organization
        for (int emp = 0; emp < 100; emp++) {
            if (emp > 0) sb.append(",\n");
            generateEmployee(sb, orgId, emp);
        }

        sb.append("\n      ]\n");
        sb.append("    }");
    }

    private static void generateEmployee(StringBuilder sb, int orgId, int empId) {
        sb.append("        {\n");
        sb.append(String.format("          \"id\": \"emp-%03d-%04d\",\n", orgId, empId));
        sb.append(String.format("          \"name\": \"Employee %d of Org %d\",\n", empId, orgId));
        sb.append(String.format("          \"email\": \"emp%d@org%d.com\",\n", empId, orgId));
        sb.append(
                String.format(
                        "          \"salary\": %.2f,\n", 50000 + RANDOM.nextDouble() * 150000));
        sb.append(String.format("          \"department\": \"dept-%d\",\n", empId % 5));

        // Skills array
        sb.append("          \"skills\": [\n");
        int numSkills = 3 + RANDOM.nextInt(5);
        for (int i = 0; i < numSkills; i++) {
            if (i > 0) sb.append(",\n");
            sb.append(String.format("            \"skill-%d\"", i));
        }
        sb.append("\n          ],\n");

        // Projects array
        sb.append("          \"projects\": [\n");
        int numProjects = 1 + RANDOM.nextInt(3);
        for (int i = 0; i < numProjects; i++) {
            if (i > 0) sb.append(",\n");
            generateProject(sb, i);
        }
        sb.append("\n          ]\n");

        sb.append("        }");
    }

    private static void generateProject(StringBuilder sb, int projId) {
        sb.append("            {\n");
        sb.append(String.format("              \"name\": \"Project %d\",\n", projId));
        sb.append(
                String.format(
                        "              \"budget\": %.2f,\n", 10000 + RANDOM.nextDouble() * 500000));
        sb.append(
                String.format(
                        "              \"start_date\": \"2024-%02d-%02d\",\n",
                        1 + RANDOM.nextInt(12), 1 + RANDOM.nextInt(28)));
        sb.append("              \"end_date\": null,\n");
        sb.append("              \"milestones\": [\n");

        int numMilestones = 2 + RANDOM.nextInt(4);
        for (int i = 0; i < numMilestones; i++) {
            if (i > 0) sb.append(",\n");
            sb.append("                {\n");
            sb.append(String.format("                  \"name\": \"Milestone %d\",\n", i));
            sb.append(String.format("                  \"completed\": %b\n", RANDOM.nextBoolean()));
            sb.append("                }");
        }

        sb.append("\n              ]\n");
        sb.append("            }");
    }
}
