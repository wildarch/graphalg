#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Operation.h>
#include <mlir/Support/LLVM.h>

#include "garel/GARelAttr.h"
#include "garel/GARelOps.h"
#include "garel/GARelTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/StringRef.h"

namespace garel {

namespace {

class SQLTranslator {
private:
  llvm::raw_ostream &_os;
  std::size_t _indentLevel = 0;

  llvm::DenseMap<mlir::Value, std::string> _valMap;
  std::size_t _tempCount;

  void indent() {
    for (auto i : llvm::seq(_indentLevel)) {
      _os << "  ";
    }
  }

  std::string newTemp() {
    return std::string("temp") + std::to_string(_tempCount++);
  }

  mlir::LogicalResult translate(mlir::func::FuncOp op);
  mlir::LogicalResult translate(mlir::Value val);
  mlir::LogicalResult translate(mlir::Operation *op);
  mlir::LogicalResult translate(ForOp op);
  mlir::LogicalResult translate(ConstantOp op);
  mlir::LogicalResult translate(AggregateOp op);
  mlir::LogicalResult translate(ProjectOp op);
  mlir::LogicalResult translate(UnionOp op);
  mlir::LogicalResult translate(JoinOp op);
  mlir::LogicalResult translate(RemapOp op);
  mlir::LogicalResult translate(SelectOp op);

  mlir::LogicalResult translate(ExtractOp op);
  mlir::LogicalResult translate(mlir::arith::SelectOp op);
  mlir::LogicalResult translate(mlir::arith::ConstantOp op);
  mlir::LogicalResult translate(mlir::arith::AddIOp op);
  mlir::LogicalResult translate(mlir::arith::AddFOp op);
  mlir::LogicalResult translate(mlir::arith::MulIOp op);
  mlir::LogicalResult translate(mlir::arith::MulFOp op);
  mlir::LogicalResult translate(mlir::arith::AndIOp op);
  mlir::LogicalResult translate(mlir::arith::OrIOp op);
  mlir::LogicalResult translate(mlir::arith::CmpIOp op);
  mlir::LogicalResult translate(mlir::arith::DivFOp op);
  mlir::LogicalResult translate(mlir::arith::SIToFPOp op);

  mlir::LogicalResult translateConstant(mlir::Location loc,
                                        mlir::Attribute attr);
  mlir::LogicalResult translateAdd(mlir::Operation *op);
  mlir::LogicalResult translateMul(mlir::Operation *op);

public:
  SQLTranslator(llvm::raw_ostream &os) : _os(os) {}

  mlir::LogicalResult translate(mlir::ModuleOp op);
};

} // namespace

mlir::LogicalResult SQLTranslator::translate(mlir::ModuleOp op) {
  for (auto &op : *op.getBody()) {
    auto funcOp = llvm::dyn_cast<mlir::func::FuncOp>(op);
    if (!funcOp) {
      return op.emitOpError("expected function");
    }

    if (mlir::failed(translate(funcOp))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(mlir::func::FuncOp op) {
  auto name = op.getSymName();
  _os << "def " << name << "(conn";
  for (auto [i, arg] : llvm::enumerate(op.getBody().getArguments())) {
    auto varName = std::string("farg") + std::to_string(i);
    _valMap[arg] = varName;
    _os << ", ";
    _os << varName;
  }

  _os << "):\n";
  _indentLevel++;

  // Visit loops first, as they cannot be done with pure SQL
  for (auto op : op.getOps<ForOp>()) {
    if (mlir::failed(translate(op))) {
      return mlir::failure();
    }
  }

  auto retOp =
      llvm::cast<mlir::func::ReturnOp>(op.getBody().front().getTerminator());
  if (retOp.getNumOperands() != 1) {
    return retOp.emitOpError("expected a single return value");
  }

  indent();
  _os << "return conn.sql(\"\"\"";
  if (mlir::failed(translate(retOp.getOperand(0)))) {
    return mlir::failure();
  }

  _os << "\"\"\")\n";
  _indentLevel--;
  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(mlir::Value val) {
  if (_valMap.contains(val)) {
    _os << "(SELECT * FROM " << _valMap[val] << ")";
    return mlir::success();
  }

  auto op = val.getDefiningOp();
  if (!op) {
    return mlir::emitError(val.getLoc())
           << val << " is not a known variable or an operation result";
  }

  return translate(op);
}

mlir::LogicalResult SQLTranslator::translate(mlir::Operation *op) {
#define CASE(OP)                                                               \
  if (auto o = llvm::dyn_cast<OP>(op)) {                                       \
    return translate(o);                                                       \
  }

  CASE(ForOp)
  CASE(ConstantOp)
  CASE(AggregateOp)
  CASE(ProjectOp)
  CASE(UnionOp)
  CASE(JoinOp)
  CASE(RemapOp)
  CASE(SelectOp)
  CASE(ExtractOp)
  CASE(mlir::arith::SelectOp)
  CASE(mlir::arith::ConstantOp)
  CASE(mlir::arith::AddIOp)
  CASE(mlir::arith::AddFOp)
  CASE(mlir::arith::MulIOp)
  CASE(mlir::arith::MulFOp)
  CASE(mlir::arith::AndIOp)
  CASE(mlir::arith::OrIOp)
  CASE(mlir::arith::CmpIOp)
  CASE(mlir::arith::DivFOp)
  CASE(mlir::arith::SIToFPOp)
#undef CASE

  return op->emitOpError("no SQL translation defined for this op");
}

mlir::LogicalResult SQLTranslator::translate(ForOp op) {
  auto &body = op.getBody().front();
  // Initialize temporary tables for loop state
  llvm::SmallVector<std::string> stateTables;
  for (auto i : llvm::seq(op.getInit().size())) {
    auto temp = newTemp();
    indent();
    _os << "conn.execute(\"\"\"CREATE TABLE " << temp << " AS ";
    if (mlir::failed(translate(op.getInit()[i]))) {
      return mlir::failure();
    }
    _os << "\"\"\")\n";

    stateTables.push_back(temp);
    _valMap[body.getArgument(i)] = temp;
  }

  indent();
  _os << "iters, = conn.sql(\"\"\"";
  if (mlir::failed(translate(op.getIters()))) {
    return mlir::failure();
  }
  _os << "\"\"\").fetchone()\n";

  indent();
  _os << "for i in range(iters):\n";
  _indentLevel++;
  auto yieldOp = llvm::cast<ForYieldOp>(op.getBody().front().getTerminator());
  llvm::SmallVector<std::string> newStateTables;
  for (auto i : llvm::seq(stateTables.size())) {
    auto temp = newTemp();
    indent();
    _os << "conn.execute(\"\"\"CREATE TABLE " << temp << " AS ";
    if (mlir::failed(translate(yieldOp.getInputs()[i]))) {
      return mlir::failure();
    }
    _os << "\"\"\")\n";

    newStateTables.push_back(temp);
  }

  // TODO: convergence check?
  // Swap to new tables
  for (auto [table, newTable] : llvm::zip_equal(stateTables, newStateTables)) {
    indent();
    _os << "conn.execute(\"DROP TABLE " << table << "\")\n";
    indent();
    _os << "conn.execute(\"ALTER TABLE " << newTable << " RENAME TO " << table
        << "\")\n";
  }

  if (!op.getUntil().empty()) {
    auto &body = op.getUntil().front();
    auto yieldOp = llvm::cast<ForYieldOp>(body.getTerminator());
    // Map block arguments
    for (auto i : llvm::seq(stateTables.size())) {
      _valMap[body.getArgument(i)] = stateTables[i];
    }

    indent();
    _os << "until, = conn.sql(\"\"\"";
    if (mlir::failed(translate(yieldOp.getInputs()[0]))) {
      return mlir::failure();
    }
    _os << "\"\"\").fetchone()\n";

    indent();
    _os << "if until:\n";
    _indentLevel++;
    indent();
    _os << "break\n";
    _indentLevel--;
  }

  _indentLevel--;

  // Bind result of the loop to state table.
  _valMap[op] = stateTables[op.getResultIdx()];

  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translateConstant(mlir::Location loc,
                                                     mlir::Attribute attr) {
  if (auto boolAttr = llvm::dyn_cast<mlir::BoolAttr>(attr)) {
    _os << (boolAttr.getValue() ? "true" : "false");
  } else if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
    _os << intAttr.getValue();
  } else if (auto floatAttr = llvm::dyn_cast<mlir::FloatAttr>(attr)) {
    auto value = floatAttr.getValue();
    if (value.isNegInfinity()) {
      _os << "'-Infinity'";
    } else if (value.isPosInfinity()) {
      _os << "'Infinity'";
    } else {
      _os << "CAST(" << value << " AS DOUBLE PRECISION)";
    }
  } else {
    return mlir::emitError(loc) << "cannot convert constant " << attr;
  }

  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(ConstantOp op) {
  _os << "(SELECT ";
  if (mlir::failed(translateConstant(op.getLoc(), op.getValue()))) {
    return mlir::failure();
  }

  _os << " AS c0)";
  return mlir::success();
}

static llvm::StringLiteral translateAggregateFunc(AggregateFunc f) {
  switch (f) {
  case AggregateFunc::SUM:
    return "SUM";
  case AggregateFunc::MIN:
    return "MIN";
  case AggregateFunc::MAX:
    return "MAX";
  case AggregateFunc::LOR:
    return "BOOL_OR";
  case AggregateFunc::ARGMIN:
    return "ARG_MIN";
  case AggregateFunc::COUNT:
    return "COUNT";
  }
}

mlir::LogicalResult SQLTranslator::translate(AggregateOp op) {
  _os << "(SELECT ";
  std::size_t colOut = 0;
  for (auto key : op.getGroupBy()) {
    if (colOut > 0) {
      _os << ", ";
    }

    _os << "c" << key << " AS c" << colOut++;
  }

  for (auto agg : op.getAggregators()) {
    if (colOut > 0) {
      _os << ", ";
    }

    _os << translateAggregateFunc(agg.getFunc()) << "(";
    llvm::interleaveComma(agg.getInputs(), _os,
                          [&](ColumnIdx idx) { _os << "c" << idx; });
    _os << ") AS c" << colOut++;
  }

  _os << " FROM ";
  if (mlir::failed(translate(op.getInput()))) {
    return mlir::failure();
  }

  if (!op.getGroupBy().empty()) {
    _os << " GROUP BY ";
    llvm::interleaveComma(op.getGroupBy(), _os,
                          [&](ColumnIdx idx) { _os << "c" << idx; });
  }

  _os << ")";
  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(ProjectOp op) {
  _os << "(SELECT ";
  auto retOp = op.getTerminator();
  for (auto [i, val] : llvm::enumerate(retOp.getProjections())) {
    if (i != 0) {
      _os << ", ";
    }

    if (mlir::failed(translate(val))) {
      return mlir::failure();
    }

    _os << " AS c" << i;
  }

  _os << " FROM ";
  if (mlir::failed(translate(op.getInput()))) {
    return mlir::failure();
  }

  _os << ")";
  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(UnionOp op) {
  if (op.getInputs().empty()) {
    return op.emitOpError("union with zero inputs");
  }

  _os << "(";
  for (auto [i, input] : llvm::enumerate(op.getInputs())) {
    if (i != 0) {
      _os << " UNION ALL ";
    }

    if (mlir::failed(translate(input))) {
      return mlir::failure();
    }
  }
  _os << ")";
  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(JoinOp op) {
  _os << "(SELECT ";
  std::size_t outIdx = 0;
  for (auto [i, input] : llvm::enumerate(op.getInputs())) {
    auto type = llvm::cast<RelationType>(input.getType());
    for (auto c : llvm::seq(type.getColumns().size())) {
      if (i != 0 || c != 0) {
        _os << ", ";
      }

      _os << "i" << i << ".c" << c << " AS c" << outIdx++;
    }
  }

  _os << " FROM ";
  for (auto [i, input] : llvm::enumerate(op.getInputs())) {
    if (i != 0) {
      _os << ", ";
    }

    _os << "(";
    if (mlir::failed(translate(input))) {
      return mlir::failure();
    }
    _os << ") i" << i;
  }

  if (!op.getPredicates().empty()) {
    _os << " WHERE ";
    for (auto [i, pred] : llvm::enumerate(op.getPredicates())) {
      if (i != 0) {
        _os << " AND ";
      }

      _os << "i" << pred.getLhsRelIdx() << ".c" << pred.getLhsColIdx() << " = "
          << "i" << pred.getRhsRelIdx() << ".c" << pred.getRhsColIdx();
    }
  }

  _os << ")";
  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(RemapOp op) {
  _os << "(SELECT ";

  ColumnIdx outIdx = 0;
  for (ColumnIdx inIdx : op.getRemap()) {
    if (outIdx != 0) {
      _os << ", ";
    }

    _os << "c" << inIdx << " AS c" << outIdx++;
  }

  _os << " FROM ";
  if (mlir::failed(translate(op.getInput()))) {
    return mlir::failure();
  }

  _os << ")";
  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(SelectOp op) {
  _os << "(SELECT * FROM ";
  if (mlir::failed(translate(op.getInput()))) {
    return mlir::failure();
  }

  _os << " WHERE ";
  auto yieldOp = op.getTerminator();
  for (auto [i, pred] : llvm::enumerate(yieldOp.getPredicates())) {
    if (i != 0) {
      _os << " AND ";
    }

    _os << "(";
    if (mlir::failed(translate(pred))) {
      return mlir::failure();
    }
    _os << ")";
  }

  _os << ")";
  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(ExtractOp op) {
  _os << "c" << op.getColumn();
  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(mlir::arith::SelectOp op) {
  _os << "(CASE WHEN ";
  if (mlir::failed(translate(op.getCondition()))) {
    return mlir::failure();
  }

  _os << " THEN ";
  if (mlir::failed(translate(op.getTrueValue()))) {
    return mlir::failure();
  }

  _os << " ELSE ";
  if (mlir::failed(translate(op.getFalseValue()))) {
    return mlir::failure();
  }

  _os << " END)";
  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(mlir::arith::ConstantOp op) {
  return translateConstant(op.getLoc(), op.getValue());
}

mlir::LogicalResult SQLTranslator::translateAdd(mlir::Operation *op) {
  _os << "(";
  if (mlir::failed(translate(op->getOperand(0)))) {
    return mlir::failure();
  }
  _os << " + ";
  if (mlir::failed(translate(op->getOperand(1)))) {
    return mlir::failure();
  }
  _os << ")";
  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(mlir::arith::AddIOp op) {
  return translateAdd(op);
}
mlir::LogicalResult SQLTranslator::translate(mlir::arith::AddFOp op) {
  return translateAdd(op);
}

mlir::LogicalResult SQLTranslator::translateMul(mlir::Operation *op) {
  _os << "(";
  if (mlir::failed(translate(op->getOperand(0)))) {
    return mlir::failure();
  }
  _os << " * ";
  if (mlir::failed(translate(op->getOperand(1)))) {
    return mlir::failure();
  }
  _os << ")";
  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(mlir::arith::MulIOp op) {
  return translateMul(op);
}
mlir::LogicalResult SQLTranslator::translate(mlir::arith::MulFOp op) {
  return translateMul(op);
}

mlir::LogicalResult SQLTranslator::translate(mlir::arith::AndIOp op) {
  _os << "(";
  if (mlir::failed(translate(op->getOperand(0)))) {
    return mlir::failure();
  }
  _os << " AND ";
  if (mlir::failed(translate(op->getOperand(1)))) {
    return mlir::failure();
  }
  _os << ")";
  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(mlir::arith::OrIOp op) {
  _os << "(";
  if (mlir::failed(translate(op->getOperand(0)))) {
    return mlir::failure();
  }
  _os << " OR ";
  if (mlir::failed(translate(op->getOperand(1)))) {
    return mlir::failure();
  }
  _os << ")";
  return mlir::success();
}

static llvm::StringLiteral translatePredicate(mlir::arith::CmpIPredicate pred) {
  switch (pred) {
  case mlir::arith::CmpIPredicate::eq:
    return "=";
  case mlir::arith::CmpIPredicate::ne:
    return "<>";
  case mlir::arith::CmpIPredicate::slt:
  case mlir::arith::CmpIPredicate::ult:
    return "<";
  case mlir::arith::CmpIPredicate::sle:
  case mlir::arith::CmpIPredicate::ule:
    return "<=";
  case mlir::arith::CmpIPredicate::sgt:
  case mlir::arith::CmpIPredicate::ugt:
    return ">";
  case mlir::arith::CmpIPredicate::sge:
  case mlir::arith::CmpIPredicate::uge:
    return ">=";
  }
}

mlir::LogicalResult SQLTranslator::translate(mlir::arith::CmpIOp op) {
  _os << "(";
  if (mlir::failed(translate(op->getOperand(0)))) {
    return mlir::failure();
  }
  _os << " " << translatePredicate(op.getPredicate()) << " ";

  if (mlir::failed(translate(op->getOperand(1)))) {
    return mlir::failure();
  }
  _os << ")";
  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(mlir::arith::DivFOp op) {
  _os << "(";
  if (mlir::failed(translate(op.getLhs()))) {
    return mlir::failure();
  }
  _os << " / ";
  if (mlir::failed(translate(op.getRhs()))) {
    return mlir::failure();
  }
  _os << ")";
  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(mlir::arith::SIToFPOp op) {
  _os << "CAST(";
  if (mlir::failed(translate(op.getIn()))) {
    return mlir::failure();
  }
  _os << " AS DOUBLE PRECISION)";
  return mlir::success();
}

mlir::LogicalResult translateToSQL(mlir::Operation *op, llvm::raw_ostream &os) {
  SQLTranslator translator(os);
  auto moduleOp = llvm::dyn_cast<mlir::ModuleOp>(op);
  if (!moduleOp) {
    return op->emitOpError("expected a module");
  }

  return translator.translate(moduleOp);
}

} // namespace garel
