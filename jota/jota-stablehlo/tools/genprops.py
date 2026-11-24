"""Generates bytecode/StablehloPropertySchemas.java from stablehlo/dialect/StablehloOps.td and
ChloOps.td.

Usage (from jota/):
  python3 jota-stablehlo/tools/genprops.py /path/to/openxla/stablehlo \
      jota-stablehlo/src/main/java/com/qxotic/jota/stablehlo/bytecode/StablehloPropertySchemas.java
  mvn -pl jota-stablehlo spotless:apply

Per StableHLO and CHLO op: its inherent attributes sorted by name, each required or optional, which is
how ODS-generated readProperties()/writeProperties() lay out the properties in bytecode
(OpDefinitionsGen.cpp: optional = OptionalAttr, a default value, or UnitAttr).

Plain bytecode is not versioned like VHLO: an op's layout changes when StableHLO adds an
attribute. Given a checkout with full history (git clone --filter=blob:none works), the
script emits the layout of every op at the last commit of each StableHLO version since
1.0.0 (versions from Version.h), one entry per change; the reader picks the layout from the
stablehlo dialect version recorded in the bytecode. A shallow checkout gives the current
layouts only.
"""
import glob
import os
import re
import subprocess
import sys

repo, out = sys.argv[1], sys.argv[2]
DIALECT = "stablehlo/dialect"

WRAPPERS = ("OptionalAttr", "DefaultValuedAttr", "DefaultValuedOptionalAttr", "ConfinedAttr",
            "DefaultValuedStrAttr", "TypedArrayAttrBase")
OPTIONAL = ("OptionalAttr", "DefaultValuedAttr", "DefaultValuedOptionalAttr",
            "DefaultValuedStrAttr", "UnitAttr")


def git(*args):
    return subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True,
                          text=True).stdout


def td_files_at(commit):
    """{file name: text} of the dialect's .td files at a commit (None: the working tree)."""
    if commit is None:
        return {os.path.basename(f): open(f).read() for f in glob.glob(f"{repo}/{DIALECT}/*.td")}
    names = git("ls-tree", "--name-only", f"{commit}:{DIALECT}/").split()
    return {n: git("show", f"{commit}:{DIALECT}/{n}") for n in names if n.endswith(".td")}


def split_top(s):
    """Splits on commas outside <>, (), [] and quotes."""
    parts, depth, cur, quote = [], 0, "", False
    for c in s:
        if c == '"':
            quote = not quote
        elif not quote and c in "<([":
            depth += 1
        elif not quote and c in ">)]":
            depth -= 1
        if c == "," and depth == 0 and not quote:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    if cur.strip():
        parts.append(cur)
    return parts


def head(t):
    m = re.match(r"\s*(\w+)", t)
    if not m:
        raise SystemExit(f"cannot classify argument type {t!r}")
    return m.group(1)


def is_optional(t, optional_defs):
    """OptionalAttr, a default value or UnitAttr, also under a wrapper (Confined<...>) or
    behind a definition that is one (StableHLO 1.0: def PrecisionConfigAttr: OptionalAttr<...>)."""
    return any(w in OPTIONAL or w in optional_defs for w in re.findall(r"\w+", t))


def layouts(files):
    """{op name: [(attribute, optional)] sorted by name} of StablehloOps.td and ChloOps.td."""
    attr_defs = set()  # every attribute definition (AttrDef, EnumAttr, ArrayOf...)
    optional_defs = set()  # definitions that are an optional attribute themselves
    for text in files.values():
        for m in re.finditer(r"^def (\w+)\s*:\s*([^{;]*)", text, re.M):
            if re.search(r"Attr|AttrDef|EnumAttr|ArrayAttr|TypedArrayAttrBase", m.group(2)):
                attr_defs.add(m.group(1))
                if head(m.group(2)) in OPTIONAL:
                    optional_defs.add(m.group(1))

    def is_attr(t):
        # any attribute word in the type: wrappers nest (Confined<I64Attr, ...>), operand
        # types (HLO_Tensor, Variadic<...>) never name an attribute
        head(t)
        words = re.findall(r"\w+", t)
        return any(w in WRAPPERS or w.endswith("Attr") or w.endswith("AttrInterface")
                   or w in attr_defs for w in words)

    result = {}
    for dialect, td_name, prefix in (("stablehlo", "StablehloOps.td", "StableHLO"),
                                     ("chlo", "ChloOps.td", "CHLO")):
        if td_name in files:
            result.update(
                dialect_layouts(files[td_name], dialect, prefix, is_attr, optional_defs))
    return result


def dialect_layouts(td, dialect, prefix, is_attr, optional_defs):
    result = {}
    for m in re.finditer(rf'^def {prefix}_\w+\s*:\s*{prefix}_\w*Op<\s*"([\w-]+)"', td, re.M):
        start = m.end()
        nxt = re.search(r"^def ", td[start:], re.M)
        body = td[start : start + nxt.start() if nxt else len(td)]
        args = re.search(r"let arguments\s*=\s*\(ins(.*?)\);", body, re.S)
        attrs = []
        if args:
            text = re.sub(r"//[^\n]*|/\*.*?\*/", "", args.group(1), flags=re.S)
            for item in split_top(text):
                m2 = re.match(r"(.*):\s*\$(\w+)\s*$", item, re.S)
                if m2 and is_attr(m2.group(1)):
                    attrs.append((m2.group(2), is_optional(m2.group(1), optional_defs)))
        result[f"{dialect}.{m.group(1)}"] = sorted(attrs)
    return result


def version_of(commit):
    text = git("show", f"{commit}:{DIALECT}/Version.h")
    m = re.search(r"getCurrentVersion\(\)\s*\{\s*return Version\((\d+),\s*(\d+),\s*(\d+)\)", text)
    return tuple(int(x) for x in m.groups()) if m else None


def history():
    """[(version, td files)]: the last commit of every version, or the working tree alone."""
    if len(git("log", "--format=%h", "-2").split()) < 2:
        return [((0, 0, 0), td_files_at(None))]
    bumps = []  # (version, first commit), oldest first
    for c in reversed(git("log", "--format=%H", "--", f"{DIALECT}/Version.h").split()):
        v = version_of(c)
        if v and (not bumps or bumps[-1][0] != v):
            bumps.append((v, c))
    snapshots = []
    for i, (v, _) in enumerate(bumps):
        last = git("rev-parse", f"{bumps[i + 1][1]}~1").strip() if i + 1 < len(bumps) else "HEAD"
        if v >= (1, 0, 0):
            snapshots.append((v, td_files_at(last)))
    return snapshots


ops = {}  # mnemonic -> [(since version, layout)], a new entry only when the layout changes
snapshots = history()
for v, files in snapshots:
    for name, attrs in layouts(files).items():
        entries = ops.setdefault(name, [])
        if not entries or entries[-1][1] != attrs:
            entries.append((v, attrs))

lines = []
for name in sorted(ops):
    for v, attrs in ops[name]:
        fields = ", ".join(f'"{a}{"?" if opt else ""}"' for a, opt in attrs)
        since = ".".join(map(str, v))
        lines.append(f'        put("{name}", "{since}"{", " + fields if fields else ""});')
first = ".".join(map(str, snapshots[0][0]))
last = ".".join(map(str, snapshots[-1][0]))

src = f'''package com.qxotic.jota.stablehlo.bytecode;

import com.qxotic.jota.stablehlo.portable.StableHloVersion;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Property layouts of the {{@code stablehlo}} and {{@code chlo}} operations, generated by {{@code
 * tools/genprops.py}} from {{@code stablehlo/dialect/StablehloOps.td}} and {{@code ChloOps.td}} at
 * every StableHLO version from {first} to {last} ({len(ops)} ops). Do not edit by hand.
 *
 * <p>The inherent attributes of an op, sorted by name, which is the order of ODS-generated {{@code
 * readProperties}}; a trailing {{@code ?}} marks an optional one (OptionalAttr, a default value,
 * UnitAttr), encoded as a flagged attribute reference. An op has one entry per layout change,
 * valid from its version until the next entry.
 */
final class StablehloPropertySchemas {{

    private record Entry(StableHloVersion since, List<PropertySchemas.Field> fields) {{}}

    private static final Map<String, List<Entry>> SCHEMAS = new HashMap<>();

    private StablehloPropertySchemas() {{}}

    private static void put(String opName, String since, String... fields) {{
        List<PropertySchemas.Field> layout =
                java.util.Arrays.stream(fields)
                        .map(
                                f ->
                                        f.endsWith("?")
                                                ? new PropertySchemas.Field(
                                                        f.substring(0, f.length() - 1), true)
                                                : new PropertySchemas.Field(f, false))
                        .toList();
        SCHEMAS.computeIfAbsent(opName, k -> new ArrayList<>())
                .add(new Entry(StableHloVersion.parse(since), layout));
    }}

    /**
     * Layout of {{@code opName}} ({{@code stablehlo.add}}, {{@code chlo.top_k}}) as written by
     * StableHLO {{@code version}} (null: the newest), or null for an unknown op.
     */
    static List<PropertySchemas.Field> get(String opName, StableHloVersion version) {{
        List<Entry> entries = SCHEMAS.get(opName);
        if (entries == null) {{
            return null;
        }}
        Entry chosen = entries.getFirst();
        for (Entry e : entries) {{
            if (version == null || e.since().compareTo(version) <= 0) {{
                chosen = e;
            }}
        }}
        return chosen.fields();
    }}

    /** Every layout of {{@code opName}}, newest first, or null. */
    static List<List<PropertySchemas.Field>> all(String opName) {{
        List<Entry> entries = SCHEMAS.get(opName);
        return entries == null
                ? null
                : entries.reversed().stream().map(Entry::fields).distinct().toList();
    }}

    static {{
{chr(10).join(lines)}
    }}
}}
'''
open(out, "w").write(src)
print(f"{len(ops)} ops, {len(lines)} layouts, versions {first} to {last} ({len(snapshots)} snapshots)")
