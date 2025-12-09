#include <charconv>
#include <iostream>
#include <optional>
#include <string_view>
#include <system_error>

#include "pg_graphalg/PgGraphAlg.h"

extern "C" {

#include <postgres.h>

#include <access/htup_details.h>
#include <access/reloptions.h>
#include <catalog/pg_proc.h>
#include <catalog/pg_type.h>
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
#include <utils/fmgrprotos.h>
#include <utils/palloc.h>
#include <utils/rel.h>
#include <utils/syscache.h>

PG_MODULE_MAGIC;

static pg_graphalg::PgGraphAlg *SINGLETON = nullptr;
static pg_graphalg::PgGraphAlg &getInstance() {
  if (!SINGLETON) {
    SINGLETON = new pg_graphalg::PgGraphAlg();
  }

  return *SINGLETON;
}

PG_FUNCTION_INFO_V1(graphalg_fdw_handler);
PG_FUNCTION_INFO_V1(graphalg_fdw_validator);
PG_FUNCTION_INFO_V1(graphalg_pl_call_handler);
PG_FUNCTION_INFO_V1(graphalg_pl_inline_handler);
PG_FUNCTION_INFO_V1(graphalg_pl_validator);

static std::optional<std::size_t> parseDimension(std::string_view s) {
  std::size_t v;
  auto res = std::from_chars(s.data(), s.data() + s.size(), v);
  if (res.ec == std::errc()) {
    return v;
  } else {
    return std::nullopt;
  }
}

static bool validateOption(DefElem *def) {
  std::string_view optName{def->defname};
  const char *optValue = defGetString(def);

  if (optName == "rows" || optName == "columns") {
    // NOTE: foreign data wrapper options are always strings.
    if (!parseDimension(optValue)) {
      ereport(ERROR, (errcode(ERRCODE_FDW_ERROR),
                      errmsg("invalid value for option \"%s\": '%s' must be "
                             "a non-negative integer",
                             def->defname, optValue)));
      return false;
    }
  } else {
    ereport(ERROR, (errcode(ERRCODE_FDW_INVALID_OPTION_NAME),
                    errmsg("invalid option \"%s\"", def->defname),
                    errhint("Valid table options are \"rows\", and "
                            "\"columns\"")));
    return false;
  }

  return true;
}

static std::optional<pg_graphalg::MatrixTableDef>
parseOptions(ForeignTable *table) {
  // Get the name of the table.
  auto relTuple = SearchSysCache1(RELOID, table->relid);
  if (!HeapTupleIsValid(relTuple)) {
    elog(ERROR, "cannot retrieve table name for oid");
    return std::nullopt;
  }

  auto relStruct = (Form_pg_class)GETSTRUCT(relTuple);
  std::string tableName{NameStr(relStruct->relname)};
  ReleaseSysCache(relTuple);

  // Parse the options
  ListCell *cell;
  std::optional<std::size_t> rows;
  std::optional<std::size_t> cols;

  foreach (cell, table->options) {
    auto *def = lfirst_node(DefElem, cell);
    std::string_view defName(def->defname);

    if (!validateOption(def)) {
      return std::nullopt;
    } else if (defName == "rows") {
      rows = parseDimension(defGetString(def));
    } else if (defName == "columns") {
      cols = parseDimension(defGetString(def));
    }
  }

  if (!rows) {
    ereport(ERROR, (errcode(ERRCODE_FDW_ERROR),
                    errmsg("missing required option \"rows\"")));
    return std::nullopt;
  }

  if (!cols) {
    ereport(ERROR, (errcode(ERRCODE_FDW_ERROR),
                    errmsg("missing required option \"columns\"")));
    return std::nullopt;
  }

  return pg_graphalg::MatrixTableDef{
      tableName,
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
  // Resolve to a matrix table.
  auto tableDef = parseOptions(GetForeignTable(foreigntableid));
  if (!tableDef) {
    return nullptr;
  }

  // Create table if it does not exist yet.
  getInstance().getOrCreateTable(foreigntableid, *tableDef);

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
  auto &table = getInstance().getTable(tableId);

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

static void BeginForeignModify(ModifyTableState *mtstate, ResultRelInfo *rinfo,
                               List *fdw_private, int subplan_index,
                               int eflags) {
  // Ensure table exists before modifying it.
  auto tableId = RelationGetRelid(rinfo->ri_RelationDesc);
  auto tableDef = parseOptions(GetForeignTable(tableId));
  if (tableDef) {
    getInstance().getOrCreateTable(tableId, *tableDef);
  }
}

static TupleTableSlot *ExecForeignInsert(EState *estate, ResultRelInfo *rinfo,
                                         TupleTableSlot *slot,
                                         TupleTableSlot *planSlot) {
  auto tableId = RelationGetRelid(rinfo->ri_RelationDesc);
  auto &table = getInstance().getTable(tableId);

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

  fdwRoutine->BeginForeignModify = BeginForeignModify;
  fdwRoutine->ExecForeignInsert = ExecForeignInsert;

  PG_RETURN_POINTER(fdwRoutine);
}

Datum graphalg_fdw_validator(PG_FUNCTION_ARGS) {
  List *options = untransformRelOptions(PG_GETARG_DATUM(0));

  ListCell *cell;
  foreach (cell, options) {
    auto *def = static_cast<DefElem *>(lfirst(cell));
    validateOption(def);
  }

  // NOTE: Not checking that required options are set, because this validator is
  // also called when checking options defined on the wrapper or the server.

  PG_RETURN_VOID();
}

static Datum executeCall(FunctionCallInfo fcinfo) {
  auto procTuple = SearchSysCache(
      PROCOID, ObjectIdGetDatum(fcinfo->flinfo->fn_oid), 0, 0, 0);
  if (!HeapTupleIsValid(procTuple)) {
    elog(ERROR, "cache lookup failed for function %s", fcinfo->flinfo->fn_oid);
    PG_RETURN_VOID();
  }

  auto procStruct = (Form_pg_proc)GETSTRUCT(procTuple);

  bool isnull;
  auto sourceDatum =
      SysCacheGetAttr(PROCOID, procTuple, Anum_pg_proc_prosrc, &isnull);
  if (isnull) {
    elog(ERROR, "NULL procedure source");
    PG_RETURN_VOID();
  }

  char *procCode = DatumGetCString(DirectFunctionCall1(textout, sourceDatum));
  std::cerr << "GraphAlg source: " << procCode << "\n";

  auto funcName = procStruct->proname.data;
  std::cerr << "function name: " << funcName << "\n";

  // TODO: Get string values of arguments.
  for (int i = 0; i < fcinfo->nargs; i++) {
    auto arg = fcinfo->args[i];
    if (arg.isnull) {
      elog(ERROR, "Argument %d is NULL", i);
      PG_RETURN_VOID();
    }

    auto argType = procStruct->proargtypes.values[i];
    auto typeTuple = SearchSysCache1(TYPEOID, ObjectIdGetDatum(argType));
    auto typeStruct = (Form_pg_type)GETSTRUCT(typeTuple);
    FmgrInfo typeInfo;
    fmgr_info(typeStruct->typoutput, &typeInfo);

    char *value = OutputFunctionCall(&typeInfo, arg.value);
    std::cerr << "arg value: " << value << "\n";

    ReleaseSysCache(typeTuple);
  }

  ReleaseSysCache(procTuple);

  elog(ERROR, "execute not implemented");
  PG_RETURN_VOID();
}

Datum graphalg_pl_call_handler(PG_FUNCTION_ARGS) {
  std::cerr << "call handler!\n";
  return executeCall(fcinfo);
}

Datum graphalg_pl_inline_handler(PG_FUNCTION_ARGS) {
  std::cerr << "inline handler!\n";
  PG_RETURN_VOID();
}

Datum graphalg_pl_validator(PG_FUNCTION_ARGS) {
  std::cerr << "validator!\n";
  PG_RETURN_VOID();
}
}
