// The command line: what the flags are, and what they mean.
package com.qxotic.jinfer.examples.inflect2;

import java.io.PrintStream;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A parsed command line. Values are range-checked here, so a failure names the flag that caused it,
 * and {@link #usage} lives beside the switch that reads them - the two describe the same thing, and
 * kept apart they drift.
 */
record Options(
        String model,
        String text,
        Path output,
        double speed,
        double variation,
        long seed,
        boolean play,
        boolean list,
        boolean help,
        Map<String, String> overrides) {

    /**
     * @throws IllegalArgumentException on anything the usage text does not describe
     */
    static Options parse(String[] args) {
        String model = null, text = "Hello world.", output = "output.wav";
        double speed = 1.0, variation = 0.667;
        long seed = 7;
        boolean play = false, list = false, help = args.length == 0;
        var overrides = new LinkedHashMap<String, String>();

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            switch (arg) {
                case "--model" -> model = value(args, ++i, arg);
                case "--text" -> text = value(args, ++i, arg);
                case "--output" -> output = value(args, ++i, arg);
                case "--speed" -> speed = number(value(args, ++i, arg), arg, 0.5, 2.0);
                case "--variation" -> variation = number(value(args, ++i, arg), arg, 0, 1);
                case "--seed" -> seed = Long.parseLong(value(args, ++i, arg));
                case "--play" -> play = true;
                case "--list" -> list = true;
                case "--help", "-h" -> help = true;
                case "--override" -> {
                    String key = value(args, ++i, arg);
                    overrides.put(key, value(args, ++i, arg));
                }
                default -> {
                    if (arg.startsWith("-"))
                        throw new IllegalArgumentException("unknown flag " + arg);
                    if (model != null)
                        throw new IllegalArgumentException("more than one model given: " + arg);
                    model = arg;
                }
            }
        }
        return new Options(
                model,
                text,
                Path.of(output),
                speed,
                variation,
                seed,
                play,
                list,
                help,
                Map.copyOf(overrides));
    }

    static void usage(PrintStream to) {
        to.println(
                """
                usage: inflect [model.gguf] [options]

                  --model <path>      model file, or z://<entry> from the executable's overlay
                  --text <string>     text to speak (default: "Hello world.")
                  --output <path>     WAV file to write (default: output.wav)
                  --speed <0.5-2.0>   speaking rate (default: 1.0)
                  --variation <0-1>   latent noise scale (default: 0.667)
                  --seed <int>        random seed (default: 7)
                  --override <k> <v>  pronunciation override, repeatable
                  --play              play it instead of writing a file (aplay/ffplay/afplay)
                  --list              list models in the executable's overlay, and their config
                  --help              show this message\
                """);
    }

    private static String value(String[] args, int index, String flag) {
        if (index >= args.length) throw new IllegalArgumentException(flag + " needs a value");
        return args[index];
    }

    private static double number(String raw, String flag, double low, double high) {
        double value;
        try {
            value = Double.parseDouble(raw);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(flag + " expects a number, got " + raw);
        }
        if (value < low || value > high)
            throw new IllegalArgumentException(flag + " must be in [" + low + "," + high + "]");
        return value;
    }
}
