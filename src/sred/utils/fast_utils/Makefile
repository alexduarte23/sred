
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    CPPFLAGS= -fPIC -shared
    OUT= fast_utils.so
endif
ifeq ($(UNAME_S),Darwin)
    CPPFLAGS= -dynamiclib -flat_namespace
    OUT= fast_utils.dylib
endif

CC= g++
SOURCES= $(filter-out dllmain.cpp, $(wildcard *.cpp))


all:
	$(CC) $(CPPFLAGS) $(SOURCES) -o $(OUT)

clean:
	rm -rf *.o $(OUT)