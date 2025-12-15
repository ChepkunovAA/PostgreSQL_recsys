EXTENSION = recsys
MODULE_big = recsys

DATA = sql/recsys--0.1.sql
OBJS = src/recsys.o src/model.o
HEADERS = src/model.h

PG_CONFIG = pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)
