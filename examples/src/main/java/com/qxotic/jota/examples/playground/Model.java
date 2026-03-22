package com.qxotic.jota.examples.playground;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Environment;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryViewPrinter;
import com.qxotic.jota.tensor.Tensor;
import java.util.ArrayList;
import java.util.List;

public interface Model<Configuration, Weights, State> {
    Configuration configuration();

    static void main(String[] args) {

        var mlp =
                new MLP(
                        "rootMLP",
                        new Linear("ppup", Tensor.of(new float[] {2}).view(Shape.of(1, 1)), null),
                        new Linear("ppgate", Tensor.of(new float[] {3}).view(Shape.of(1, 1)), null),
                        new Linear(
                                "ppdown", Tensor.of(new float[] {5}).view(Shape.of(1, 1)), null));

        Tensor result = mlp.forward(Tensor.of(new float[] {7}).view(Shape.of(1, 1)), null);
        MemoryAccess memoryAccess =
                Environment.nativeRuntime().memoryDomain().directAccess();
        String string = MemoryViewPrinter.toString(result.materialize(), memoryAccess);
        System.out.println(string);

        mlp.accept(
                new ModuleVisitor() {

                    List<String> pp = new ArrayList<>();

                    @Override
                    public void enterModule(String path, Module<?, ?> module) {
                        pp.addLast(path);
                    }

                    @Override
                    public void exitModule(String path, Module<?, ?> module) {
                        pp.removeLast();
                    }

                    @Override
                    public void visitTensor(String path, Tensor tensor) {
                        System.out.println(String.join(".", pp) + " -> " + tensor);
                    }
                });
    }
}

interface ModuleContext {}

final class RecordReflection {

    public static void accept(Record recordModule, ModuleVisitor visitor) {
        Module<?, ?> m = (Module<?, ?>) recordModule;
        String name = m.name();
        visitor.enterModule(name, m);
        acceptRecord(recordModule, name, visitor);
        visitor.exitModule(name, m);
    }

    private static void acceptRecord(Record record, String prefix, ModuleVisitor visitor) {
        var components = record.getClass().getRecordComponents();

        for (var component : components) {
            String compName = component.getName();

            // Skip "name" component - it's metadata, not a parameter
            if (compName.equals("name")) {
                continue;
            }

            String path = prefix.isEmpty() ? compName : prefix + "." + compName;

            try {
                Object value = component.getAccessor().invoke(record);
                visitValue(path, value, visitor);
            } catch (ReflectiveOperationException e) {
                throw new RuntimeException("Failed to access component: " + compName, e);
            }
        }
    }

    private static void visitValue(String path, Object value, ModuleVisitor visitor) {
        if (value == null) {
            return;
        }

        switch (value) {
            // Tensor
            case Tensor t -> visitor.visitTensor(path, t);

            // Nested Module
            case Module<?, ?> m -> {
                visitor.enterModule(path, m);
                m.accept(visitor);
                visitor.exitModule(path, m);
            }

            // Primitives (boxed)
            case Boolean b -> visitor.visitPrimitive(path, b ? 1 : 0, DataType.BOOL);
            case Byte b -> visitor.visitPrimitive(path, b, DataType.I8);
            case Short s -> visitor.visitPrimitive(path, s, DataType.I16);
            case Integer i -> visitor.visitPrimitive(path, i, DataType.I32);
            case Long l -> visitor.visitPrimitive(path, l, DataType.I64);
            case Float f -> visitor.visitPrimitive(path, Float.floatToRawIntBits(f), DataType.FP32);
            case Double d ->
                    visitor.visitPrimitive(path, Double.doubleToRawLongBits(d), DataType.FP64);

            case String s -> visitor.visitString(path, s);

            // Nested record (not a Module)
            default -> throw new UnsupportedOperationException();
        }
    }
}

interface Module<I, O> {
    O forward(I input, ModuleContext ctx);

    String name();

    default void accept(ModuleVisitor visitor) {
        if (this instanceof Record recordModule) {
            RecordReflection.accept(recordModule, visitor);
        } else {
            throw new UnsupportedOperationException("Module is not a record");
        }
    }
}

// Simple visitor interface
interface ModuleVisitor {
    default void enterModule(String path, Module<?, ?> module) {}

    default void exitModule(String path, Module<?, ?> module) {}

    default void visitTensor(String path, Tensor tensor) {}

    default void visitPrimitive(String path, long rawBits, DataType dataType) {}

    default void visitString(String path, String value) {}
}

record Linear(String name, Tensor weight, Tensor bias) implements Module<Tensor, Tensor> {
    @Override
    public Tensor forward(Tensor x, ModuleContext ctx) {
        var y = weight.matmul(x);
        if (bias == null) {
            return y;
        } else {
            return y.add(bias.broadcast(y.shape()));
        }
    }
}

record MLP(String name, Linear gateProj, Linear upProj, Linear downProj)
        implements Module<Tensor, Tensor> {
    @Override
    public Tensor forward(Tensor x, ModuleContext ctx) {
        var gate = gateProj.forward(x, ctx).silu();
        var up = upProj.forward(x, ctx);
        return downProj.forward(gate.multiply(up), ctx);
    }

    // No accept() needed - recursively visits gateProj, upProj, downProj!
}
