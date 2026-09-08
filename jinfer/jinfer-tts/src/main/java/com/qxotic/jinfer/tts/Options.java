package com.qxotic.jinfer.tts;

import java.io.PrintStream;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;

record Options(
        String model,
        Map<String, String> companions,
        String text,
        Path output,
        Double speed,
        boolean play,
        boolean stream,
        boolean list,
        boolean help,
        Path archive) {

    static Options parse(String[] args) {
        String model = null, text = "Hello world.", output = "output.wav";
        Double speed = null;
        boolean play = false, stream = false, list = false, help = args.length == 0;
        Path archive = null;
        Map<String, String> companions = new LinkedHashMap<>();
        for (int i = 0; i < args.length; i++) {
            String arg = args[i];
            switch (arg) {
                case "--model" -> model = value(args, ++i, arg);
                case "--with" -> {
                    String attached = value(args, ++i, arg);
                    int equals = attached.indexOf('=');
                    if (equals < 1 || equals == attached.length() - 1)
                        throw new IllegalArgumentException("--with expects <name>=<source>");
                    String previous =
                            companions.put(
                                    attached.substring(0, equals), attached.substring(equals + 1));
                    if (previous != null)
                        throw new IllegalArgumentException(
                                "companion given twice: " + attached.substring(0, equals));
                }
                case "--text" -> text = value(args, ++i, arg);
                case "--output" -> output = value(args, ++i, arg);
                case "--speed" -> {
                    String raw = value(args, ++i, arg);
                    try {
                        speed = Double.parseDouble(raw);
                    } catch (NumberFormatException e) {
                        throw new IllegalArgumentException("--speed expects a number, got " + raw);
                    }
                    if (!Double.isFinite(speed) || speed <= 0)
                        throw new IllegalArgumentException("--speed must be positive and finite");
                }
                case "--play" -> play = true;
                case "--stream" -> stream = true;
                case "--list" -> list = true;
                case "--archive" -> archive = Path.of(value(args, ++i, arg));
                case "--help", "-h" -> help = true;
                default -> {
                    if (arg.startsWith("-"))
                        throw new IllegalArgumentException("unknown flag " + arg);
                    if (model != null)
                        throw new IllegalArgumentException("more than one model given: " + arg);
                    model = arg;
                }
            }
        }
        if (play && stream)
            throw new IllegalArgumentException("--play and --stream cannot be used together");
        return new Options(
                model,
                Map.copyOf(companions),
                text,
                Path.of(output),
                speed,
                play,
                stream,
                list,
                help,
                archive);
    }

    static void usage(PrintStream to) {
        to.println(
                """
                Usage: jinfer-tts [MODEL] [OPTIONS]

                  MODEL                   GGUF file or z:// archive entry
                  --model SOURCE          same as positional MODEL
                  --with NAME=SOURCE      companion such as voice or lexicon (repeatable)
                  --text TEXT             text to speak (default: "Hello world.")
                  --speed RATE            speaking-rate multiplier
                  --output FILE           write WAV output (default: output.wav)
                  --play                  synthesize, then play
                  --stream                play each clip as soon as it is ready
                  --archive FILE          read z:// entries from FILE instead of this executable
                  --list                  list usable archive entries
                  -h, --help              show this help\
                """);
    }

    private static String value(String[] args, int index, String flag) {
        if (index >= args.length) throw new IllegalArgumentException(flag + " needs a value");
        return args[index];
    }
}
