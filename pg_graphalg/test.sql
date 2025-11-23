DROP FOREIGN TABLE IF EXISTS mat;
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

CREATE FOREIGN TABLE mat ( row bigint, col bigint, val bigint ) SERVER graphalg_server;
SELECT * FROM mat;

INSERT INTO mat VALUES
  (0, 0, 42),
  (0, 1, 43),
  (1, 0, 44),
  (1, 1, 45);

SELECT * FROM mat;

INSERT INTO mat VALUES
  (0, 1, 4000);

SELECT * FROM mat;
