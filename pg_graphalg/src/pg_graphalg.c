#include <postgres.h>

#include <executor/tuptable.h>
#include <fmgr.h>
#include <foreign/fdwapi.h>
#include <nodes/pg_list.h>
#include <optimizer/pathnode.h>
#include <optimizer/planmain.h>
#include <optimizer/restrictinfo.h>
#include <utils/palloc.h>

PG_MODULE_MAGIC;

PG_FUNCTION_INFO_V1(add_one);

PG_FUNCTION_INFO_V1(graphalg_fdw_handler);

Datum add_one(PG_FUNCTION_ARGS) {
  int32 arg = PG_GETARG_INT32(0);

  PG_RETURN_INT32(arg + 1);
}

static void GetForeignRelSize(PlannerInfo *root, RelOptInfo *baserel,
                              Oid foreigntableid) {
  baserel->rows = 42;
}

static void GetForeignPaths(PlannerInfo *root, RelOptInfo *baserel,
                            Oid foreigntableid) {
  ForeignPath *path = create_foreignscan_path(root, baserel,
                                              /*target=*/NULL,
                                              /*rows=*/baserel->rows,
                                              /*disabled_nodes=*/0,
                                              /*startup_cost=*/1,
                                              /*total_cost=*/1 + baserel->rows,
                                              /*pathkeys=*/NIL,
                                              /*required_outer=*/NULL,
                                              /*fdw_outerpath=*/NULL,
                                              /*fdw_restrictinfo=*/NULL,
                                              /*fdw_private=*/NULL);
  add_path(baserel, (Path *)path);
}

static ForeignScan *GetForeignPlan(PlannerInfo *root, RelOptInfo *baserel,
                                   Oid foreigntableid, ForeignPath *best_path,
                                   List *tlist, List *scan_clauses,
                                   Plan *outer_plan) {
  // On extract_actual_clauses:
  // https://www.postgresql.org/docs/current/fdw-planning.html
  scan_clauses = extract_actual_clauses(scan_clauses, false);
  return make_foreignscan(
      tlist, scan_clauses, baserel->relid,
      NIL, /* no expressions we will evaluate */
      NIL, /* no private data */
      NIL, /* no custom tlist; our scan tuple looks like tlist */
      NIL, /* no quals we will recheck */
      outer_plan);
}

typedef struct {
  int current;
} GaScanState;

static void BeginForeignScan(ForeignScanState *node, int eflags) {
  node->fdw_state = palloc0(sizeof(GaScanState));
}

static TupleTableSlot *IterateForeignScan(ForeignScanState *node) {
  TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
  ExecClearTuple(slot);

  GaScanState *state = (GaScanState *)node->fdw_state;
  if (state->current < 10) {
    slot->tts_isnull[0] = false;
    slot->tts_values[0] = Int32GetDatum(state->current);
    ExecStoreVirtualTuple(slot);
    state->current++;
  }

  return slot;
}

static void ReScanForeignScan(ForeignScanState *node) {
  GaScanState *state = (GaScanState *)node->fdw_state;
  state->current = 0;
}

static void EndForeignScan(ForeignScanState *node) {
  // No-Op
}

static TupleTableSlot *ExecForeignInsert(EState *estate, ResultRelInfo *rinfo,
                                         TupleTableSlot *slot,
                                         TupleTableSlot *planSlot) {
  // TODO: Actually save it somewhere.
  bool isnull;
  Datum datum = slot_getattr(slot, 1, &isnull);
  if (isnull) {
    printf("NULL\n");
  } else {
    int x = DatumGetInt32(datum);
    printf("value: %d\n", x);
  }

  return slot;
}

Datum graphalg_fdw_handler(PG_FUNCTION_ARGS) {
  FdwRoutine *fdwRoutine = makeNode(FdwRoutine);

  fdwRoutine->GetForeignRelSize = GetForeignRelSize;
  fdwRoutine->GetForeignPaths = GetForeignPaths;
  fdwRoutine->GetForeignPlan = GetForeignPlan;
  fdwRoutine->BeginForeignScan = BeginForeignScan;
  fdwRoutine->IterateForeignScan = IterateForeignScan;
  fdwRoutine->ReScanForeignScan = ReScanForeignScan;
  fdwRoutine->EndForeignScan = EndForeignScan;

  fdwRoutine->ExecForeignInsert = ExecForeignInsert;

  PG_RETURN_POINTER(fdwRoutine);
}
