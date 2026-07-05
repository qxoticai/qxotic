/* Windows (win32) jni_md.h for cross-compiling the JNI binding from Linux.
 * The JDK ships this only inside a Windows distribution (include/win32/); it is machine-dependent but
 * OS-specific (identical for windows-x86-64 and windows-aarch64), so this one file serves both. Standard
 * OpenJDK contents. Pair with the JDK's include/jni.h via -DJNI_INCLUDE_DIRS. */
#ifndef _JAVASOFT_JNI_MD_H_
#define _JAVASOFT_JNI_MD_H_

#define JNIEXPORT __declspec(dllexport)
#define JNIIMPORT __declspec(dllimport)
#define JNICALL __stdcall

typedef long jint;
typedef long long jlong;
typedef signed char jbyte;

#endif /* !_JAVASOFT_JNI_MD_H_ */
