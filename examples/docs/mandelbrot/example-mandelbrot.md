
Running it..

com.qxotic.jota.examples.demos.Mandelbrot

using CPU only using 'panama', on Linux
```text
WARNING: A terminally deprecated method in sun.misc.Unsafe has been called
WARNING: sun.misc.Unsafe::allocateMemory has been called by com.qxotic.jota.memory.internal.NativeUnsafeAllocator (file:/data/qxoticai/qxotic/jota/jota-memory/target/classes/)
WARNING: Please consider reporting this to the maintainers of class com.qxotic.jota.memory.internal.NativeUnsafeAllocator
WARNING: sun.misc.Unsafe::allocateMemory will be removed in a future release
WARNING: A restricted method in java.lang.foreign.MemorySegment has been called
WARNING: java.lang.foreign.MemorySegment::reinterpret has been called by com.qxotic.jota.memory.internal.NativeUnsafeAllocator$RawScopedMemory in an unnamed module (file:/data/qxoticai/qxotic/jota/jota-memory/target/classes/)
WARNING: Use --enable-native-access=ALL-UNNAMED to avoid a warning for callers in this module
WARNING: Restricted methods will be blocked in a future release unless native access is enabled

Backend=panama, runtime=3141ms (800x600, MAX_ITER=100)
Writing Mandelbrot (iter=100) result to file 'mandelbrot-panama.ppm'
```

![mandelbrot-panama.png](mandelbrot-panama.png)
