package com.qxotic.jota.tensor;

import com.qxotic.jota.memory.MemoryView;
import java.util.List;
import java.util.Map;

sealed interface LazyComputation
        permits ConstantComputation, RangeComputation, IRComputation, RandomComputation {

    List<Tensor> inputs();

    Map<String, Object> attributes();

    MemoryView<?> execute();
}
