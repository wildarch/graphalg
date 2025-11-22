MODULE_big = tutorial_fdw
OBJS = tutorial_fdw.o

EXTENSION = tutorial_fdw
DATA = tutorial_fdw--1.0.sql

PG_CONFIG = /usr/local/pgsql/bin/pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)