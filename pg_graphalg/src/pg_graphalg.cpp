#include <optional>
#include <string_view>

#include "pg_graphalg/PgGraphAlg.h"

extern "C" {

#include <postgres.h>

#include <commands/defrem.h>
#include <executor/tuptable.h>
#include <fmgr.h>
#include <foreign/fdwapi.h>
#include <foreign/foreign.h>
#include <nodes/parsenodes.h>
#include <nodes/pg_list.h>
#include <optimizer/pathnode.h>
#include <optimizer/planmain.h>
#include <optimizer/restrictinfo.h>
#include <utils/elog.h>
#include <utils/palloc.h>
#include <utils/rel.h>

PG_MODULE_MAGIC;

static pg_graphalg::PgGraphAlg *SINGLETON = nullptr;
static pg_graphalg::PgGraphAlg &getInstance() {
  if (!SINGLETON) {
    SINGLETON = new pg_graphalg::PgGraphAlg();
  }

  return *SINGLETON;
}

PG_FUNCTION_INFO_V1(graphalg_fdw_handler);

static std::optional<pg_graphalg::MatrixTableDef>
parseOptions(ForeignTable *table) {
  ListCell *cell;
  std::optional<std::int64_t> rows = 0;
  std::optional<std::int64_t> cols = 0;

  foreach (cell, table->options) {
    auto *def = lfirst_node(DefElem, cell);
    std::string_view defName(def->defname);
    if (defName == "rows") {
      // TODO: Check type of option
      rows = defGetInt64(def);
      if (rows < 0) {
        ereport(ERROR, (errcode(ERRCODE_FDW_ERROR),
                        errmsg("invalid value for option \"rows\": %d must be "
                               "a positive integer",
                               *rows)));
        return std::nullopt;
      }
    } else if (defName == "columns") {
      // TODO: Check type of option
      cols = defGetInt64(def);
      if (rows < 0) {
        ereport(ERROR, (errcode(ERRCODE_FDW_ERROR),
                        errmsg("invalid value for option \"cols\": %d must be "
                               "a positive integer",
                               *cols)));
        return std::nullopt;
      }
    } else {
      ereport(ERROR,
              (errcode(ERRCODE_FDW_INVALID_OPTION_NAME),
               errmsg("invalid option \"%s\"", def->defname),
               errhint("Valid table options for graphalg are \"rows\", and "
                       "\"columns\"")));
      return std::nullopt;
    }
  }

  if (!rows) {
    ereport(ERROR, (errcode(ERRCODE_FDW_ERROR),
                    errmsg("missing required option \"rows\"")));
    return std::nullopt;
  }

  return pg_graphalg::MatrixTableDef{
      static_cast<size_t>(*rows),
      static_cast<size_t>(*cols),
  };
}

static void GetForeignRelSize(PlannerInfo *root, RelOptInfo *baserel,
                              Oid foreigntableid) {
  auto tableDef = parseOptions(GetForeignTable(foreigntableid));
  if (!tableDef) {
    return;
  }

  auto &table = getInstance().getOrCreateTable(foreigntableid, *tableDef);
  baserel->rows = table.nValues();
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

static void BeginForeignScan(ForeignScanState *node, int eflags) {
  auto tableId = RelationGetRelid(node->ss.ss_currentRelation);
  // TODO: Avoid parsing options multiple times?
  auto tableDef = parseOptions(GetForeignTable(tableId));
  if (!tableDef) {
    return;
  }

  auto &table = getInstance().getOrCreateTable(tableId, *tableDef);

  auto *state = palloc(sizeof(pg_graphalg::ScanState));
  new (state) pg_graphalg::ScanState(&table);
  node->fdw_state = state;
}

static TupleTableSlot *IterateForeignScan(ForeignScanState *node) {
  TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
  ExecClearTuple(slot);

  auto *scanState = static_cast<pg_graphalg::ScanState *>(node->fdw_state);
  auto &table = *scanState->table;
  if (auto res = table.scan(*scanState)) {
    slot->tts_isnull[0] = false;
    slot->tts_isnull[1] = false;
    slot->tts_isnull[2] = false;
    auto [row, col, val] = *res;
    slot->tts_values[0] = UInt64GetDatum(row);
    slot->tts_values[1] = UInt64GetDatum(col);
    slot->tts_values[2] = UInt64GetDatum(val);
    ExecStoreVirtualTuple(slot);
  }

  return slot;
}

static void ReScanForeignScan(ForeignScanState *node) {
  auto *scanState = static_cast<pg_graphalg::ScanState *>(node->fdw_state);
  scanState->reset();
}

static void EndForeignScan(ForeignScanState *node) {
  // No-Op
}

static TupleTableSlot *ExecForeignInsert(EState *estate, ResultRelInfo *rinfo,
                                         TupleTableSlot *slot,
                                         TupleTableSlot *planSlot) {
  auto tableId = RelationGetRelid(rinfo->ri_RelationDesc);
  // TODO: Avoid parsing options multiple times?
  auto tableDef = parseOptions(GetForeignTable(tableId));
  if (!tableDef) {
    return NULL;
  }

  auto &table = getInstance().getOrCreateTable(tableId, *tableDef);

  slot_getsomeattrs(slot, 3);
  if (slot->tts_isnull[0] || slot->tts_isnull[1] || slot->tts_isnull[2]) {
    // Ignore nulls
    return nullptr;
  }

  std::size_t row = DatumGetUInt64(slot->tts_values[0]);
  std::size_t col = DatumGetUInt64(slot->tts_values[1]);
  std::int64_t val = DatumGetInt64(slot->tts_values[2]);
  table.setValue(row, col, val);
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
}
