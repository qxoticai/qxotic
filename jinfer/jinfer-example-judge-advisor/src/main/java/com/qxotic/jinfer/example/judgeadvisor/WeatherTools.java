package com.qxotic.jinfer.example.judgeadvisor;

import java.util.ArrayDeque;
import java.util.List;
import java.util.Queue;
import org.springframework.ai.tool.annotation.Tool;

/**
 * The weather gimmick from the original demo, made deterministic: a cycling queue of readings.
 * -255C is below absolute zero - physically impossible, so the judge must fail it; 15C passes.
 */
public final class WeatherTools {

    private final Queue<Integer> readings;

    public WeatherTools() {
        this.readings = new ArrayDeque<>(List.of(-255, 15));
    }

    @Tool(description = "Get the current weather for a given location")
    public String weather(String location) {
        int temperature = readings.poll();
        readings.add(temperature); // cycle forever
        System.out.printf(">>> tool: weather(%s) -> %dC%n", location, temperature);
        return "The current weather in "
                + location
                + " is sunny with a temperature of "
                + temperature
                + "C.";
    }
}
