// A loaded GGUF tensor: name, GGML quant type, shape, and its mmap'd data segment. Public so the
// jinfer-gemma4 loader can consume ModelLoader's tensor maps.
package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;
import java.lang.foreign.MemorySegment;

public record GGMLTensorEntry(
        String name, GGMLType ggmlType, int[] shape, MemorySegment memorySegment) {}
