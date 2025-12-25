BREW ?= brew
FAISS_PREFIX ?= $(shell $(BREW) --prefix faiss 2>/dev/null)
MYSQL_PREFIX ?= $(shell $(BREW) --prefix mysql-client 2>/dev/null)
OMP_PREFIX ?= $(shell $(BREW) --prefix libomp 2>/dev/null)

CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17
LDFLAGS ?=
LDLIBS ?= -lfaiss -lmysqlclient

ifneq ($(FAISS_PREFIX),)
  CXXFLAGS += -I$(FAISS_PREFIX)/include
  LDFLAGS += -L$(FAISS_PREFIX)/lib
endif

CXXFLAGS += -Ihnswlib/hnswlib

ifneq ($(MYSQL_PREFIX),)
  CXXFLAGS += -I$(MYSQL_PREFIX)/include
  LDFLAGS += -L$(MYSQL_PREFIX)/lib
endif

ifneq ($(OMP_PREFIX),)
  CXXFLAGS += -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
  LDFLAGS += -L$(OMP_PREFIX)/lib
  LDLIBS += -lomp
endif

TARGET = knn_compare
all: $(TARGET)

$(TARGET): main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ $^ $(LDLIBS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGET) main.o

.PHONY: all clean
