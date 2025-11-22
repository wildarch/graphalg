#include "postgres.h"

#include "access/table.h"
#include "fmgr.h"
#include "foreign/fdwapi.h"
#include "optimizer/pathnode.h"
#include "optimizer/planmain.h"
#include "optimizer/restrictinfo.h"
#include "utils/rel.h"

Datum tutorial_fdw_handler(PG_FUNCTION_ARGS);

PG_FUNCTION_INFO_V1(tutorial_fdw_handler);

void tutorial_fdw_GetForeignRelSize(PlannerInfo *root, RelOptInfo *baserel,
                                    Oid foreigntableid);

void tutorial_fdw_GetForeignPaths(PlannerInfo *root, RelOptInfo *baserel,
                                  Oid foreigntableid);

ForeignScan *tutorial_fdw_GetForeignPlan(PlannerInfo *root, RelOptInfo *baserel,
                                         Oid foreigntableid,
                                         ForeignPath *best_path, List *tlist,
                                         List *scan_clauses, Plan *outer_plan);

void tutorial_fdw_BeginForeignScan(ForeignScanState *node, int eflags);

TupleTableSlot *tutorial_fdw_IterateForeignScan(ForeignScanState *node);

void tutorial_fdw_ReScanForeignScan(ForeignScanState *node);

void tutorial_fdw_EndForeignScan(ForeignScanState *node);

Datum tutorial_fdw_handler(PG_FUNCTION_ARGS) {
  FdwRoutine *fdwroutine = makeNode(FdwRoutine);
  fdwroutine->GetForeignRelSize = tutorial_fdw_GetForeignRelSize;

  fdwroutine->GetForeignPaths = tutorial_fdw_GetForeignPaths;

  fdwroutine->GetForeignPlan = tutorial_fdw_GetForeignPlan;

  fdwroutine->BeginForeignScan = tutorial_fdw_BeginForeignScan;

  fdwroutine->IterateForeignScan = tutorial_fdw_IterateForeignScan;

  fdwroutine->ReScanForeignScan = tutorial_fdw_ReScanForeignScan;

  fdwroutine->EndForeignScan = tutorial_fdw_EndForeignScan;

  PG_RETURN_POINTER(fdwroutine);
}

void tutorial_fdw_GetForeignRelSize(PlannerInfo *root, RelOptInfo *baserel,
                                    Oid foreigntableid) {
  Relation rel = table_open(foreigntableid, NoLock);

  if (rel->rd_att->natts != 1) {

    ereport(ERROR,

            errcode(ERRCODE_FDW_INVALID_COLUMN_NUMBER),

            errmsg("incorrect schema for tutorial_fdw table %s: table must "
                   "have exactly one column",
                   NameStr(rel->rd_rel->relname)));
  }

  /*
  Oid typid = rel->rd_att->attrs[0].atttypid;

  if (typid != INT4OID) {

    ereport(ERROR,

            errcode(ERRCODE_FDW_INVALID_DATA_TYPE),

            errmsg("incorrect schema for tutorial_fdw table %s: table column "
                   "must have type int",
                   NameStr(rel->rd_rel->relname)));
  }
  */

  table_close(rel, NoLock);
}

void tutorial_fdw_GetForeignPaths(PlannerInfo *root, RelOptInfo *baserel,
                                  Oid foreigntableid) {
  Path *path = (Path *)create_foreignscan_path(
      root, baserel, NULL, /* default pathtarget */
      baserel->rows,       /* rows */
      0,                   /* disabled_nodes */
      1,                   /* startup cost */
      1 + baserel->rows,   /* total cost */
      NIL,                 /* no pathkeys */
      NULL,                /* no required outer relids */
      NULL,                /* no fdw_outerpath */
      NULL,                /* no fdw_restrictinfo */
      NIL);                /* no fdw_private */

  add_path(baserel, path);
}

ForeignScan *tutorial_fdw_GetForeignPlan(PlannerInfo *root, RelOptInfo *baserel,
                                         Oid foreigntableid,

                                         ForeignPath *best_path, List *tlist,
                                         List *scan_clauses, Plan *outer_plan) {
  scan_clauses = extract_actual_clauses(scan_clauses, false);
  return make_foreignscan(
      tlist, scan_clauses, baserel->relid,
      NIL, /* no expressions we will evaluate */
      NIL, /* no private data */
      NIL, /* no custom tlist; our scan tuple looks like tlist */
      NIL, /* no quals we will recheck */
      outer_plan);
}

typedef struct tutorial_fdw_state {

  int current;

} tutorial_fdw_state;

void tutorial_fdw_BeginForeignScan(ForeignScanState *node, int eflags) {
  tutorial_fdw_state *state = palloc0(sizeof(tutorial_fdw_state));

  node->fdw_state = state;
}

TupleTableSlot *tutorial_fdw_IterateForeignScan(ForeignScanState *node) {
  TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;

  ExecClearTuple(slot);

  tutorial_fdw_state *state = node->fdw_state;

  if (state->current < 64) {

    slot->tts_isnull[0] = false;

    slot->tts_values[0] = Int32GetDatum(state->current);

    ExecStoreVirtualTuple(slot);

    state->current++;
  }

  return slot;
}

void tutorial_fdw_ReScanForeignScan(ForeignScanState *node) {
  tutorial_fdw_state *state = node->fdw_state;

  state->current = 0;
}

void tutorial_fdw_EndForeignScan(ForeignScanState *node) {}

PG_MODULE_MAGIC;
