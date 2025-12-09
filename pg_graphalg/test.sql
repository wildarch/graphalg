DROP FOREIGN TABLE IF EXISTS mat1;
DROP FOREIGN TABLE IF EXISTS mat2;
DROP FOREIGN TABLE IF EXISTS lhs;
DROP FOREIGN TABLE IF EXISTS rhs;
DROP FOREIGN TABLE IF EXISTS matmul_out;
DROP SERVER IF EXISTS graphalg_server;
DROP FOREIGN DATA WRAPPER IF EXISTS graphalg_fdw;
DROP FUNCTION IF EXISTS graphalg_fdw_handler;
DROP FUNCTION IF EXISTS graphalg_fdw_validator;

DROP PROCEDURE IF EXISTS matmul;
DROP LANGUAGE IF EXISTS graphalg;
DROP FUNCTION IF EXISTS graphalg_pl_call_handler;
DROP FUNCTION IF EXISTS graphalg_pl_inline_handler;
DROP FUNCTION IF EXISTS graphalg_pl_validator;

-- Foreign data wrapper
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

-- Procedural Language
CREATE FUNCTION graphalg_pl_call_handler()
RETURNS language_handler
AS '/workspaces/graphalg/pg_graphalg/build/src/libpg_graphalg.so'
LANGUAGE C STRICT;

CREATE FUNCTION graphalg_pl_validator(oid) RETURNS void
AS '/workspaces/graphalg/pg_graphalg/build/src/libpg_graphalg.so'
LANGUAGE C STRICT;

CREATE FUNCTION graphalg_pl_inline_handler(internal)
RETURNS void
AS '/workspaces/graphalg/pg_graphalg/build/src/libpg_graphalg.so'
LANGUAGE C STRICT;

CREATE TRUSTED LANGUAGE graphalg
HANDLER graphalg_pl_call_handler
INLINE graphalg_pl_inline_handler
VALIDATOR graphalg_pl_validator;

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

CREATE FOREIGN TABLE lhs(row bigint, col bigint, val bigint)
SERVER graphalg_server
OPTIONS (rows '10', columns '10');

INSERT INTO lhs VALUES
  (0, 0, 42),
  (0, 1, 43),
  (1, 0, 44),
  (1, 1, 45);
CREATE FOREIGN TABLE rhs(row bigint, col bigint, val bigint)
SERVER graphalg_server
OPTIONS (rows '10', columns '10');
INSERT INTO rhs VALUES
  (0, 0, 46),
  (0, 1, 47),
  (1, 0, 48),
  (1, 1, 49);

CREATE FOREIGN TABLE matmul_out(row bigint, col bigint, val bigint)
SERVER graphalg_server
OPTIONS (rows '10', columns '10');
-- HACK: Necessary because we don't get a callback from CREATE
SELECT * FROM matmul_out;

CREATE PROCEDURE matmul(text, text, text)
LANGUAGE graphalg
AS $$
  func matmul(
      lhs: Matrix<s, s, int>,
      rhs: Matrix<s, s, int>) -> Matrix<s, s, int> {
    return lhs * rhs;
  }
$$;

CALL matmul('lhs', 'rhs', 'matmul_out');
