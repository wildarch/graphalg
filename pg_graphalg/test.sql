CREATE OR REPLACE FUNCTION add_one(integer) RETURNS integer
     AS '/workspaces/graphalg/pg_graphalg/build/src/libpg_graphalg.so', 'add_one'
     LANGUAGE C STRICT;

SELECT add_one(42);
