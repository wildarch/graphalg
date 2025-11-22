#!/bin/bash

set -eo pipefail

make
#sudo make install

PGDATA=`mktemp -d -t tfdw-XXXXXXXXXXX`

trap "PGDATA=\"$PGDATA\" pg_ctl stop >/dev/null || true; rm -rf \"$PGDATA\"" EXIT

PGDATA="$PGDATA" pg_ctl initdb > /dev/null
PGDATA="$PGDATA" pg_ctl start
psql postgres -f smoke_test.sql