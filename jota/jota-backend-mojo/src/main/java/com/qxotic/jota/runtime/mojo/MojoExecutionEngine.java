package com.qxotic.jota.runtime.mojo;

import com.qxotic.jota.runtime.KernelCacheKey;
import com.qxotic.jota.runtime.KernelExecutable;
import com.qxotic.jota.runtime.KernelProgram;
import com.qxotic.jota.runtime.KernelService;

/** Minimal execution API used by the Mojo frontend. */
interface MojoExecutionEngine<T> {

    MojoMemoryDomain<T> memoryDomain();

    long addressOf(T pointer);

    KernelService kernelService();

    KernelExecutable getOrCompile(KernelProgram program, KernelCacheKey key);
}
