CC ?= gcc
CFLAGS ?= -O2 -fPIC -std=gnu17 -D_GNU_SOURCE
LDFLAGS ?= -shared -ldl

JAVA_HOME ?= /usr/lib/jvm/java-25-graalvm
JNI_INCLUDE = -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux

TARGET = libjota_c.so
SRC = jota_c_jni.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(JNI_INCLUDE) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
