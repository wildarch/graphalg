-- Create the matrix table for the graph
CREATE FOREIGN TABLE graph(source bigint, target bigint, dist double precision)
SERVER graphalg_server OPTIONS (rows '10', columns '10');
INSERT INTO graph VALUES
    (0, 1, 0.5),
    (0, 2, 5.0),
    (0, 3, 5.0),
    (1, 4, 0.5),
    (2, 3, 2.0),
    (4, 5, 0.5),
    (5, 2, 0.5),
    (5, 9, 23.0),
    (6, 0, 1.0),
    (6, 7, 3.2),
    (7, 9, 0.2),
    (8, 9, 0.1),
    (9, 6, 8.0);

-- Define the algorithm
CREATE PROCEDURE SSSP(text, text, text)
LANGUAGE graphalg
AS $$
func sssp(
    graph: Matrix<s, s, trop_real>,
    source: Vector<s, trop_real>) -> Vector<s, trop_real> {
  dist = source;
  for i in graph.nrows {
    dist += dist * graph;
  }
  return dist;
}
$$;

-- Start from source node 0
CREATE FOREIGN TABLE source(vertex_id bigint, nop bigint, init_dist double precision)
SERVER graphalg_server OPTIONS (rows '10', columns '1');
INSERT INTO source VALUES (0, 0, 0.0);

-- Output of the algorithm
CREATE FOREIGN TABLE dist_out(vertex_id bigint, nop bigint, dist double precision)
SERVER graphalg_server OPTIONS (rows '10', columns '1');

-- Run he algorithm
CALL SSSP('graph', 'source', 'dist_out');

-- Read the results.
SELECT vertex_id, dist FROM dist_out;

DROP FOREIGN TABLE graph;
DROP FOREIGN TABLE source;
DROP FOREIGN TABLE dist_out;
DROP PROCEDURE SSSP;
