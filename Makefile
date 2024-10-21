# -std=c++14: we're limiting ourselves to c++14, since that's what the
#             GCC compiler on the VSC supports.
# -DNDEBUG: turns off e.g. assertion checks
# -O3: enables optimizations in the compiler

# Settings for optimized build
FLAGS= -std=c++14 -O3 -DNDEBUG
# Settings for a debug build
#FLAGS=-g -std=c++14


SRC_DIR=src
MAIN=$(SRC_DIR)/main_startcode.cpp

all: kmeans

clean:
	rm -f kmeans

kmeans: $(MAIN) $(SRC_DIR)/rng.cpp $(SRC_DIR)/kmeans_impl.hpp
	$(CXX) $(FLAGS) -o kmeans $(MAIN) $(SRC_DIR)/rng.cpp
