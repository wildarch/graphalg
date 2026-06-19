#include <charconv>
#include <optional>
#include <string_view>
#include <variant>

#include <llvm/ADT/Sequence.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Diagnostics.h>

#include "pg_graphalg/MatrixTable.h"
#include "pg_graphalg/PgGraphAlg.h"

extern "C" {

#include <postgres.h>

#include <access/htup.h>
#include <access/htup_details.h>
#include <access/reloptions.h>
#include <access/tupdesc.h>
#include <catalog/pg_proc.h>
#include <catalog/pg_type.h>
#include <commands/defrem.h>
#include <executor/spi.h>
#include <executor/tuptable.h>
#include <fmgr.h>
#include <foreign/fdwapi.h>
#include <foreign/foreign.h>
#include <nodes/parsenodes.h>
#include <nodes/pg_list.h>
#include <optimizer/pathnode.h>
#include <optimizer/planmain.h>
#include <optimizer/restrictinfo.h>
#include <postgres_ext.h>
#include <utils/elog.h>
#include <utils/fmgrprotos.h>
#include <utils/palloc.h>
#include <utils/rel.h>
#include <utils/relcache.h>
#include <utils/syscache.h>

PG_MODULE_MAGIC;

static void diagHandler(mlir::Diagnostic &diag) {
  std::string msg = diag.str();
  switch (diag.getSeverity()) {
  case mlir::DiagnosticSeverity::Note:
  case mlir::DiagnosticSeverity::Remark:
    elog(INFO, "%s", msg.c_str());
    break;
  case mlir::DiagnosticSeverity::Warning:
    elog(WARNING, "%s", msg.c_str());
    break;
  case mlir::DiagnosticSeverity::Error:
    elog(ERROR, "%s", msg.c_str());
    break;
  }
}

static pg_graphalg::PgGraphAlg *SINGLETON = nullptr;
static pg_graphalg::PgGraphAlg &getInstance() {
  if (!SINGLETON) {
    SINGLETON = new pg_graphalg::PgGraphAlg(diagHandler);
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

static std::optional<pg_graphalg::MatrixValueType> mapValueType(Oid typeId) {
  switch (typeId) {
  case BOOLOID:
    return pg_graphalg::MatrixValueType::BOOL;
  case INT8OID:
    return pg_graphalg::MatrixValueType::INT;
  case FLOAT8OID:
    return pg_graphalg::MatrixValueType::FLOAT;
  default:
    return std::nullopt;
  }
}

struct SysCacheTupleScope {
  HeapTuple tup;

  SysCacheTupleScope(SysCacheIdentifier cacheId, Oid key)
      : tup(SearchSysCache1(cacheId, key)) {}
  ~SysCacheTupleScope() {
    if (HeapTupleIsValid(tup)) {
      ReleaseSysCache(tup);
    }
  }
};

struct RelationScope {
  Relation rel;

  RelationScope(Oid relid) : rel(RelationIdGetRelation(relid)) {}
  ~RelationScope() { RelationClose(rel); }
};

static std::optional<pg_graphalg::MatrixTableDef> lookupMatrixTable(Oid relid) {
  // Must be a foreign table
  // TODO: Check that it uses a graphalg server.
  auto *fTable = GetForeignTable(relid);
  if (!fTable) {
    elog(ERROR, "relation with oid %u is not a foreign table", relid);
    return std::nullopt;
  }

  RelationScope rel(relid);
  std::string tableName{NameStr(rel.rel->rd_rel->relname)};

  // Validate the column types.
  int nAttrs = RelationGetNumberOfAttributes(rel.rel);
  if (nAttrs != 3) {
    elog(ERROR, "matrix table must have 3 columns, got %d", nAttrs);
    return std::nullopt;
  }

  TupleDesc tupleDesc = rel.rel->rd_att;
  auto rowAttr = TupleDescAttr(tupleDesc, 0);
  if (rowAttr->atttypid != INT8OID) {
    elog(ERROR, "first column (row index) must have type bigint");
    return std::nullopt;
  }

  auto colAttr = TupleDescAttr(tupleDesc, 1);
  if (colAttr->atttypid != INT8OID) {
    elog(ERROR, "second column (column index) must have type bigint");
    return std::nullopt;
  }

  auto valAttr = TupleDescAttr(tupleDesc, 2);
  auto valType = mapValueType(valAttr->atttypid);
  if (!valType) {
    elog(ERROR, "third column (value) must have type boolean, bigint or double "
                "precision");
    return std::nullopt;
  }

  // Parse the options
  ListCell *cell;
  std::optional<std::size_t> rows;
  std::optional<std::size_t> cols;

  foreach (cell, fTable->options) {
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
      *valType,
  };
}

static void GetForeignRelSize(PlannerInfo *root, RelOptInfo *baserel,
                              Oid foreigntableid) {
  auto table =
      getInstance().getOrCreateTable(foreigntableid, lookupMatrixTable);
  if (table) {
    baserel->rows = (*table)->nValues();
  }
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
  auto table = getInstance().getOrCreateTable(tableId, lookupMatrixTable);
  if (!table) {
    return;
  }

  auto *state = palloc(sizeof(pg_graphalg::MatrixTableScanState));
  new (state) pg_graphalg::MatrixTableScanState(*table);
  node->fdw_state = state;
}

static Datum matrixValueGetDatum(std::variant<bool, std::int64_t, double> v) {
  if (auto *b = std::get_if<bool>(&v)) {
    return BoolGetDatum(*b);
  } else if (auto *i = std::get_if<std::int64_t>(&v)) {
    return Int64GetDatum(*i);
  } else {
    return Float8GetDatum(std::get<double>(v));
  }
}

static std::variant<bool, std::int64_t, double>
datumGetMatrixValue(pg_graphalg::MatrixValueType type, Datum v) {
  switch (type) {
  case pg_graphalg::MatrixValueType::BOOL:
    return DatumGetBool(v);
  case pg_graphalg::MatrixValueType::INT:
    return DatumGetInt64(v);
  case pg_graphalg::MatrixValueType::FLOAT:
    return DatumGetFloat8(v);
  }
}

static TupleTableSlot *IterateForeignScan(ForeignScanState *node) {
  TupleTableSlot *slot = node->ss.ss_ScanTupleSlot;
  ExecClearTuple(slot);

  auto *scanState =
      static_cast<pg_graphalg::MatrixTableScanState *>(node->fdw_state);
  auto &table = *scanState->table;
  if (auto res = table.scan(*scanState)) {
    slot->tts_isnull[0] = false;
    slot->tts_isnull[1] = false;
    slot->tts_isnull[2] = false;
    auto [row, col, val] = *res;
    slot->tts_values[0] = UInt64GetDatum(row);
    slot->tts_values[1] = UInt64GetDatum(col);
    slot->tts_values[2] = matrixValueGetDatum(val);
    ExecStoreVirtualTuple(slot);
  }

  return slot;
}

static void ReScanForeignScan(ForeignScanState *node) {
  auto *scanState =
      static_cast<pg_graphalg::MatrixTableScanState *>(node->fdw_state);
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
  getInstance().getOrCreateTable(tableId, lookupMatrixTable);
}

static TupleTableSlot *ExecForeignInsert(EState *estate, ResultRelInfo *rinfo,
                                         TupleTableSlot *slot,
                                         TupleTableSlot *planSlot) {
  auto tableId = RelationGetRelid(rinfo->ri_RelationDesc);
  auto &table = **getInstance().getOrCreateTable(tableId, lookupMatrixTable);

  slot_getsomeattrs(slot, 3);
  if (slot->tts_isnull[0] || slot->tts_isnull[1] || slot->tts_isnull[2]) {
    // Ignore nulls
    return nullptr;
  }

  std::size_t row = DatumGetUInt64(slot->tts_values[0]);
  std::size_t col = DatumGetUInt64(slot->tts_values[1]);
  auto val = datumGetMatrixValue(table.getType(), slot->tts_values[2]);
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
    // TODO: Only allow options at the table level.
    validateOption(def);
  }

  // NOTE: Not checking that required options are set, because this validator is
  // also called when checking options defined on the wrapper or the server.

  PG_RETURN_VOID();
}

// NOTE: Assumes a working SPI connection
static std::optional<Oid> lookupForeignTable(Oid argType, Datum argValue) {
  constexpr bool READ_ONLY = true;
  /*
   * Allow up to 2 results.
   * n == 0: ERROR Table not found
   * n  > 1: ERROR Multiple oids for the given name
   * n == 1: OK Name uniquely identifies table
   */
  constexpr long TCOUNT = 2;
  int execRes = SPI_execute_with_args(
      "SELECT oid FROM pg_class WHERE relname=$1 AND relkind = 'f'", 1,
      &argType, &argValue, nullptr, READ_ONLY, TCOUNT);
  if (execRes < 0) {
    elog(ERROR, "internal error finding argument tables");
    PG_RETURN_VOID();
  }

  // Expect exactly one result.
  if (SPI_processed != 1) {
    // TODO: We know this is a string, so can we make this simpler?
    auto typeTuple = SearchSysCache1(TYPEOID, ObjectIdGetDatum(argType));
    auto typeStruct = (Form_pg_type)GETSTRUCT(typeTuple);
    FmgrInfo typeInfo;
    fmgr_info(typeStruct->typoutput, &typeInfo);
    char *value = OutputFunctionCall(&typeInfo, argValue);
    ReleaseSysCache(typeTuple);

    if (SPI_processed == 0) {
      elog(ERROR, "no such matrix table '%s'", value);
    } else {
      elog(ERROR, "multiple tables named '%s'", value);
    }

    PG_RETURN_VOID();
  }

  auto *tuptable = SPI_tuptable;
  auto tupdesc = tuptable->tupdesc;
  bool oidNull = false;
  auto tableOidDatum =
      SPI_getbinval(tuptable->vals[0], tuptable->tupdesc, 1, &oidNull);
  return DatumGetObjectId(tableOidDatum);
}

/** Automatically calls SPI_finish() when it goes out of scope. */
class SPIConnection {
public:
  SPIConnection() { SPI_connect(); }
  ~SPIConnection() { SPI_finish(); }
};

static Datum executeCall(FunctionCallInfo fcinfo) {
  auto procTuple = SearchSysCache(
      PROCOID, ObjectIdGetDatum(fcinfo->flinfo->fn_oid), 0, 0, 0);
  if (!HeapTupleIsValid(procTuple)) {
    elog(ERROR, "cache lookup failed for function %s", fcinfo->flinfo->fn_oid);
    PG_RETURN_VOID();
  }

  auto procStruct = (Form_pg_proc)GETSTRUCT(procTuple);

  // Extract program source code
  bool sourceIsNull;
  auto sourceDatum =
      SysCacheGetAttr(PROCOID, procTuple, Anum_pg_proc_prosrc, &sourceIsNull);
  if (sourceIsNull) {
    elog(ERROR, "NULL procedure source");
    PG_RETURN_VOID();
  }

  char *procCode = DatumGetCString(DirectFunctionCall1(textout, sourceDatum));

  // Get argument matrix tables.
  SPIConnection spiConnection;
  llvm::SmallVector<pg_graphalg::MatrixTable *> arguments;
  for (int i = 0; i < fcinfo->nargs; i++) {
    auto arg = fcinfo->args[i];
    if (arg.isnull) {
      elog(ERROR, "Argument %d is NULL", i);
      PG_RETURN_VOID();
    }

    auto argType = procStruct->proargtypes.values[i];
    auto tableOid = lookupForeignTable(argType, arg.value);
    if (!tableOid) {
      PG_RETURN_VOID();
    }

    auto table = getInstance().getOrCreateTable(*tableOid, lookupMatrixTable);
    if (!table) {
      PG_RETURN_VOID();
    }

    arguments.push_back(*table);
  }

  if (arguments.empty()) {
    elog(ERROR, "must have at least one argument");
    PG_RETURN_VOID();
  }

  // Output is written to the final procedure argument.
  auto *output = arguments.pop_back_val();

  // No need to check the result here, postgres infers success based on
  // diagnostics.
  auto funcName = procStruct->proname.data;
  getInstance().execute(procCode, funcName, arguments, *output);

  ReleaseSysCache(procTuple);
  PG_RETURN_VOID();
}

Datum graphalg_pl_call_handler(PG_FUNCTION_ARGS) { return executeCall(fcinfo); }

Datum graphalg_pl_inline_handler(PG_FUNCTION_ARGS) {
  elog(ERROR, "inline handler not implemented");
  PG_RETURN_VOID();
}

Datum graphalg_pl_validator(PG_FUNCTION_ARGS) {
  elog(INFO, "NOTE: language validator not implemented");
  PG_RETURN_VOID();
}
}
