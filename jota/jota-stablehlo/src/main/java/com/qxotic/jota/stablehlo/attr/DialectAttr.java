package com.qxotic.jota.stablehlo.attr;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

/**
 * Dialect attribute {@code #dialect.mnemonic<...>}. When the body is a list of {@code key = value}
 * parameters (e.g. {@code #stablehlo.gather<offset_dims = [1], index_vector_dim = 1>}) they are
 * parsed into {@link #params()}; otherwise ({@code #stablehlo.conv<[b, 0, f]x...>}) the body is
 * kept verbatim in {@link #rawBody()}.
 */
public record DialectAttr(
        String dialect, String mnemonic, List<NamedAttribute> params, String rawBody)
        implements Attribute {

    public DialectAttr {
        Objects.requireNonNull(dialect, "dialect");
        Objects.requireNonNull(mnemonic, "mnemonic");
        if ((params == null) == (rawBody == null)) {
            throw new IllegalArgumentException("exactly one of params / rawBody must be set");
        }
        params = params == null ? null : List.copyOf(params);
    }

    public static DialectAttr ofParams(
            String dialect, String mnemonic, List<NamedAttribute> params) {
        return new DialectAttr(dialect, mnemonic, params, null);
    }

    public static DialectAttr ofRaw(String dialect, String mnemonic, String rawBody) {
        return new DialectAttr(dialect, mnemonic, null, rawBody);
    }

    public boolean hasParams() {
        return params != null;
    }

    public Optional<Attribute> param(String name) {
        if (params == null) {
            return Optional.empty();
        }
        for (NamedAttribute p : params) {
            if (p.name().equals(name)) {
                return Optional.of(p.value());
            }
        }
        return Optional.empty();
    }

    /**
     * Struct attributes are keyed by parameter name: two attributes with the same parameters in a
     * different order are equal (MLIR prints them in declaration order, but parses any order).
     */
    @Override
    public boolean equals(Object o) {
        if (!(o instanceof DialectAttr other)) {
            return false;
        }
        if (!dialect.equals(other.dialect) || !mnemonic.equals(other.mnemonic)) {
            return false;
        }
        if (params == null || other.params == null) {
            return java.util.Objects.equals(rawBody, other.rawBody)
                    && java.util.Objects.equals(params, other.params);
        }
        return paramMap().equals(other.paramMap());
    }

    @Override
    public int hashCode() {
        return java.util.Objects.hash(dialect, mnemonic, params == null ? rawBody : paramMap());
    }

    private java.util.Map<String, Attribute> paramMap() {
        java.util.Map<String, Attribute> m = new java.util.HashMap<>();
        for (NamedAttribute p : params) {
            m.put(p.name(), p.value());
        }
        return m;
    }

    /** Parameter values print without their types: {@code [1, 2]}, {@code 3}, {@code "s"}. */
    static String paramLiteral(Attribute value) {
        return switch (value) {
            case IntegerAttr i -> i.literal();
            case FloatAttr f -> f.literal();
            case DenseArrayAttr a -> "[" + a.elementsLiteral() + "]";
            case ArrayAttr a -> {
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < a.elements().size(); i++) {
                    if (i > 0) {
                        sb.append(", ");
                    }
                    sb.append(paramLiteral(a.elements().get(i)));
                }
                yield sb.append(']').toString();
            }
            case DictionaryAttr d -> { // anonymous nested struct: <key = value, ...>
                StringBuilder sb = new StringBuilder("<");
                for (int i = 0; i < d.entries().size(); i++) {
                    if (i > 0) {
                        sb.append(", ");
                    }
                    NamedAttribute e = d.entries().get(i);
                    sb.append(e.name()).append(" = ").append(paramLiteral(e.value()));
                }
                yield sb.append('>').toString();
            }
            case DialectAttr s
                    when s.dialect.equals("stablehlo")
                            && s.mnemonic.equals("sub_axis_info")
                            && s.hasParams() -> // (pre_size)size
                    "("
                            + paramLiteral(s.param("pre_size").orElseThrow())
                            + ")"
                            + paramLiteral(s.param("size").orElseThrow());
            default -> value.toString();
        };
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("#").append(dialect).append('.').append(mnemonic);
        sb.append('<');
        if (params != null) {
            for (int i = 0; i < params.size(); i++) {
                if (i > 0) {
                    sb.append(", ");
                }
                sb.append(params.get(i).name())
                        .append(" = ")
                        .append(paramLiteral(params.get(i).value()));
            }
        } else {
            sb.append(rawBody);
        }
        return sb.append('>').toString();
    }
}
