CREATE EXTENSION tutorial_fdw;
CREATE SERVER tutorial_server FOREIGN DATA WRAPPER tutorial_fdw;
CREATE FOREIGN TABLE sequential_ints ( val int ) SERVER tutorial_server;
SELECT * FROM sequential_ints;