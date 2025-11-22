DROP FOREIGN TABLE IF EXISTS sequential_ints;
DROP SERVER IF EXISTS graphalg_server;
DROP FOREIGN DATA WRAPPER IF EXISTS graphalg_fdw;
DROP FUNCTION IF EXISTS graphalg_fdw_handler;

CREATE FUNCTION graphalg_fdw_handler()
RETURNS fdw_handler
AS '/workspaces/graphalg/pg_graphalg/build/src/libpg_graphalg.so'
LANGUAGE C STRICT;

CREATE FOREIGN DATA WRAPPER graphalg_fdw
  HANDLER graphalg_fdw_handler;
CREATE SERVER graphalg_server FOREIGN DATA WRAPPER graphalg_fdw;

CREATE FOREIGN TABLE sequential_ints ( val int ) SERVER graphalg_server;
SELECT * FROM sequential_ints;
