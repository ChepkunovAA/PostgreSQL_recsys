EXTENSION = recsys
MODULE_big = recsys

DATA = sql/recsys--0.1.sql sql/recsys--0.1--0.2.sql
OBJS = src/recsys.o src/transformer.o src/esasrec.o src/manager.o 
HEADERS = src/transformer.h src/esasrec.h src/manager.h

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)

LIBTORCH_DIR = /tmp/recsys/lib/libtorch

# LibTorch includes
TORCH_INCLUDES = -I$(LIBTORCH_DIR)/include
TORCH_INCLUDES += -I$(LIBTORCH_DIR)/include/torch/csrc/api/include

# LibTorch libraries 
TORCH_LIBS = -L$(LIBTORCH_DIR)/lib
TORCH_LIBS += -Wl,-rpath,$(LIBTORCH_DIR)/lib
TORCH_LIBS += -Wl,--no-as-needed
TORCH_LIBS += -ltorch_cpu -ltorch -lc10
TORCH_LIBS += -Wl,--as-needed

# Compiler flags
PG_CPPFLAGS += $(TORCH_INCLUDES) -std=c++17 -fPIC -D_GLIBCXX_USE_CXX11_ABI=1

# Linker flags
SHLIB_LINK += $(TORCH_LIBS) -lpthread -lstdc++ -lm

include $(PGXS)
