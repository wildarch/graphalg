# GraphAlg in Postgres
We need to write a postgres C extension to add GraphAlg support to it.
Resources:
- https://www.postgresql.org/docs/current/xfunc-c.html
- https://www.pgedge.com/blog/introduction-to-postgres-extension-development
- https://stackoverflow.com/questions/76056209/postgresql-c-extension-function-table-as-argument-and-as-result

## Build from source
Download the latest postgres release source code.
```bash
cd thirdparty/
wget https://ftp.postgresql.org/pub/source/v18.1/postgresql-18.1.tar.bz2
```

Following https://www.postgresql.org/docs/current/install-make.html

Packages to install:
- bison
- flex
- libreadline-dev
- libicu-dev

```bash
./configure
make
sudo make install
```

Now the binaries are located at `/usr/local/pgsql/bin`.

## Setup a database
```bash
export LC_ALL=C
export LC_CTYPE=C

# Create a new DB
/usr/local/pgsql/bin/initdb -D ~/pgdata

# Start the server
/usr/local/pgsql/bin/postgres -D ~/pgdata

```

## Load extension
First build with `pgext/build.sh`.

Then connect to the server with `/usr/local/pgsql/bin/psql postgres` and run:

```bash
CREATE FUNCTION add_one(integer) RETURNS integer
     AS '/workspaces/graphalg/pgext/funcs', 'add_one'
     LANGUAGE C STRICT;
```

## Foreign Data Wrapper
Resources:
- https://www.postgresql.org/docs/current/fdwhandler.html
- https://www.dolthub.com/blog/2022-01-26-creating-a-postgres-foreign-data-wrapper/
- https://github.com/Kentik-Archive/wdb_fdw
