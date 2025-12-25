BREW ?= $(shell command -v brew 2>/dev/null)
FAISS_PREFIX ?= $(shell if [ -n "$(BREW)" ]; then HOMEBREW_NO_AUTO_UPDATE=1 $(BREW) --prefix faiss 2>/dev/null; fi)
FAISS_DIR ?= faiss
FAISS_BUILD ?= $(FAISS_DIR)/build
FAISS_SUBMODULE := $(wildcard $(FAISS_DIR)/CMakeLists.txt)
FAISS_BUILD_TYPE ?= Release
FAISS_OPT_LEVEL ?=
FAISS_OPT_LEVEL_DETECT ?= scripts/detect_faiss_opt_level.sh
FAISS_VERBOSE ?= 0
MYSQL_PREFIX ?= $(shell if [ -n "$(BREW)" ]; then HOMEBREW_NO_AUTO_UPDATE=1 $(BREW) --prefix mysql-client 2>/dev/null; fi)
OMP_PREFIX ?= $(shell if [ -n "$(BREW)" ]; then HOMEBREW_NO_AUTO_UPDATE=1 $(BREW) --prefix libomp 2>/dev/null; fi)
UNAME_S := $(shell uname -s)

CXX ?= g++
CXXFLAGS ?= -O2 -std=c++17
LDFLAGS ?=
LDLIBS ?= -lfaiss -lmysqlclient

ifneq ($(FAISS_SUBMODULE),)
  CXXFLAGS += -I$(FAISS_DIR)
  LDFLAGS += -L$(FAISS_BUILD)/faiss -L$(FAISS_BUILD)
else ifneq ($(FAISS_PREFIX),)
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
else ifeq ($(UNAME_S),Linux)
  CXXFLAGS += -fopenmp
  LDLIBS += -fopenmp
endif

ifeq ($(UNAME_S),Darwin)
  LDLIBS += -framework Accelerate
else ifeq ($(UNAME_S),Linux)
  ifneq ($(FAISS_SUBMODULE),)
    LDLIBS += -lblas -llapack
  endif
endif

ifneq ($(FAISS_SUBMODULE),)
FAISS_BUILD_TARGET := faiss
endif

TARGET = knn_compare
all: $(TARGET)

$(TARGET): main.o $(FAISS_BUILD_TARGET)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ main.o $(LDLIBS)

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

clean:
	rm -f $(TARGET) main.o

ifneq ($(FAISS_SUBMODULE),)
faiss: $(FAISS_SUBMODULE)
	@echo "Building FAISS submodule (this may take a while)..."
	@opt_level="$(FAISS_OPT_LEVEL)"; \
	if [ -z "$$opt_level" ]; then \
	  opt_level="$$(sh $(FAISS_OPT_LEVEL_DETECT))"; \
	fi; \
	echo "Using FAISS_OPT_LEVEL=$$opt_level"; \
	cmake -B $(FAISS_BUILD) -S $(FAISS_DIR) \
		-DCMAKE_CXX_COMPILER=$(CXX) \
		-DCMAKE_BUILD_TYPE=$(FAISS_BUILD_TYPE) \
		-DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_GPU=OFF \
		-DFAISS_OPT_LEVEL=$$opt_level \
		-DBUILD_TESTING=OFF -DFAISS_ENABLE_PERF_TESTS=OFF \
		$(if $(OMP_PREFIX),-DOpenMP_ROOT=$(OMP_PREFIX),)
	cmake --build $(FAISS_BUILD) $(if $(filter 1 true yes,$(FAISS_VERBOSE)),--verbose,)
endif

.PHONY: all clean faiss
