# pg_graphalg: Run GraphAlg in PostgreSQL
This is an extension for PostgreSQL to execute GraphAlg programs.

## Building
In addition to setting up the devcontainer for the project, you need to:
- Build postgreSQL v18 from source and install it to the default path `/usr/local/pgsql`.
- Setup a dummy database in `~/pgdata`
- Install SuiteSparse:GraphBLAS

### PostgreSQL
Download the latest postgres release source code.

```bash
# Dependencies for building PostgreSQL
sudo apt update
sudo apt install bison flex libreadline-dev libicu-dev

# Download the sources
mkdir -p thirdparty/
cd thirdparty/
wget https://ftp.postgresql.org/pub/source/v18.1/postgresql-18.1.tar.bz2
tar xf postgresql-18.1.tar.bz2
cd postgresql-18.1/

# Build and install
./configure
make
sudo make install
```

Now PostgreSQL is installed at `/usr/local/pgsql`.

### Setup a database
```bash
# Need to set or PostgreSQL will refuse to create the DB
export LC_ALL=C
export LC_CTYPE=C

# Create a new DB
/usr/local/pgsql/bin/initdb -D ~/pgdata
```

### GraphBLAS
Install using APT:

```bash
sudo apt install libgraphblas-dev
```

### Build the Extension
```bash
pg_graphalg/configure.sh

# Extension is located at pg_graphalg/build/src/libpg_graphalg.so
cmake --build pg_graphalg/build
```

## Testing
After the extension has been built, start the server and then run the `test.sql` script:

```bash
# Start the server
/usr/local/pgsql/bin/postgres -D ~/pgdata

# Run the tests
/usr/local/pgsql/bin/psql postgres -f pg_graphalg/test/test.sql
```
