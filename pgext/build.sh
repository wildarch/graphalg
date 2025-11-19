#!/bin/bash

PG_SERVER_INCLUDE_DIR=$(/usr/local/pgsql/bin/pg_config --includedir-server)

gcc -I $PG_SERVER_INCLUDE_DIR -fPIC -c pgext/funcs.c -o pgext/funcs.o
gcc -shared -o pgext/funcs.so pgext/funcs.o

