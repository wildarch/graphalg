DROP FOREIGN TABLE IF EXISTS mat1;
DROP FOREIGN TABLE IF EXISTS mat2;
DROP SERVER IF EXISTS graphalg_server;
DROP FOREIGN DATA WRAPPER IF EXISTS graphalg_fdw;
DROP FUNCTION IF EXISTS graphalg_fdw_handler;
DROP FUNCTION IF EXISTS graphalg_fdw_validator;

CREATE FUNCTION graphalg_fdw_handler()
RETURNS fdw_handler
AS '/workspaces/graphalg/pg_graphalg/build/src/libpg_graphalg.so'
LANGUAGE C STRICT;

CREATE FUNCTION graphalg_fdw_validator(text[], oid)
RETURNS void
AS '/workspaces/graphalg/pg_graphalg/build/src/libpg_graphalg.so'
LANGUAGE C STRICT;

CREATE FOREIGN DATA WRAPPER graphalg_fdw
  HANDLER graphalg_fdw_handler
  VALIDATOR graphalg_fdw_validator;
CREATE SERVER graphalg_server FOREIGN DATA WRAPPER graphalg_fdw;

CREATE FOREIGN TABLE mat1 ( row bigint, col bigint, val bigint ) SERVER graphalg_server OPTIONS (rows '10', columns '10');
CREATE FOREIGN TABLE mat2 ( row bigint, col bigint, val bigint ) SERVER graphalg_server OPTIONS (rows '100', columns '100');
SELECT * FROM mat1;
SELECT * FROM mat2;

INSERT INTO mat1 VALUES
  (0, 0, 42),
  (0, 1, 43),
  (1, 0, 44),
  (1, 1, 45);

INSERT INTO mat2 VALUES
  (0, 0, 420),
  (0, 1, 430),
  (1, 0, 440),
  (1, 1, 450);

SELECT * FROM mat1;

SELECT * FROM mat2;

INSERT INTO mat1 VALUES
  (0, 1, 4000);

SELECT * FROM mat1;
