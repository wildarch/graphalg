CREATE FUNCTION tutorial_fdw_handler()
RETURNS fdw_handler
AS '/workspaces/graphalg/pgext/tutorial_fdw/tutorial_fdw'
LANGUAGE C STRICT;

CREATE FOREIGN DATA WRAPPER tutorial_fdw
  HANDLER tutorial_fdw_handler;
