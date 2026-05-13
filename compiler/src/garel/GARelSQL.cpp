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

namespace garel {

namespace {

class SQLTranslator {
private:
  llvm::raw_ostream &_os;
  llvm::DenseMap<mlir::Value, std::string> _valMap;
  std::size_t _tempCount;

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

  mlir::LogicalResult translate(ExtractOp op);
  mlir::LogicalResult translate(mlir::arith::SelectOp op);
  mlir::LogicalResult translate(mlir::arith::ConstantOp op);
  mlir::LogicalResult translate(mlir::arith::AddIOp op);
  mlir::LogicalResult translate(mlir::arith::AddFOp op);
  mlir::LogicalResult translate(mlir::arith::MulIOp op);
  mlir::LogicalResult translate(mlir::arith::MulFOp op);

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
  _os << "def " << name << "(";
  for (auto [i, arg] : llvm::enumerate(op.getBody().getArguments())) {
    auto varName = std::string("farg") + std::to_string(i);
    _valMap[arg] = varName;

    if (i != 0) {
      _os << ", ";
    }

    _os << varName;
  }

  _os << "):\n";

  auto retOp =
      llvm::cast<mlir::func::ReturnOp>(op.getBody().front().getTerminator());
  if (retOp.getNumOperands() != 1) {
    return retOp.emitOpError("expected a single return value");
  }

  return translate(retOp.getOperand(0));
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
  CASE(ExtractOp)
  CASE(mlir::arith::SelectOp)
  CASE(mlir::arith::ConstantOp)
  CASE(mlir::arith::AddIOp)
  CASE(mlir::arith::AddFOp)
  CASE(mlir::arith::MulIOp)
  CASE(mlir::arith::MulFOp)
#undef CASE

  return op->emitOpError("no SQL translation defined for this op");
}

mlir::LogicalResult SQLTranslator::translate(ForOp op) {
  auto &body = op.getBody().front();
  // Initialize temporary tables for loop state
  llvm::SmallVector<std::string> stateTables;
  for (auto i : llvm::seq(op.getInit().size())) {
    auto temp = newTemp();
    _os << "CREATE TABLE " << temp << " AS ";
    if (mlir::failed(translate(op.getInit()[i]))) {
      return mlir::failure();
    }
    _os << ";\n";

    stateTables.push_back(temp);
    _valMap[body.getArgument(i)] = temp;
  }

  _os << "iters = ";
  if (mlir::failed(translate(op.getIters()))) {
    return mlir::failure();
  }
  _os << "\n";

  _os << "for i in iters:\n";
  auto yieldOp = llvm::cast<ForYieldOp>(op.getBody().front().getTerminator());
  llvm::SmallVector<std::string> newStateTables;
  for (auto i : llvm::seq(stateTables.size())) {
    auto temp = newTemp();
    _os << "  CREATE TABLE " << temp << " AS (";
    if (mlir::failed(translate(yieldOp.getInputs()[i]))) {
      return mlir::failure();
    }
    _os << ");\n";

    newStateTables.push_back(temp);
  }

  if (!op.getUntil().empty()) {
    return op.emitOpError("'until' not implemented");
  }

  // TODO: convergence check?
  // Swap to new tables
  for (auto [table, newTable] : llvm::zip_equal(stateTables, newStateTables)) {
    _os << "  DROP TABLE " << table << ";\n";
    _os << "  ALTER TABLE " << newTable << " RENAME TO " << table << ";\n";
  }

  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(ConstantOp op) {
  _os << "(SELECT " << op.getValue() << " AS c0)";
  return mlir::success();
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

    _os << stringifyAggregateFunc(agg.getFunc()) << "(";
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

  _os << ")";
  return mlir::success();
}

mlir::LogicalResult SQLTranslator::translate(mlir::arith::ConstantOp op) {
  _os << op.getValue();
  return mlir::success();
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

mlir::LogicalResult translateToSQL(mlir::Operation *op, llvm::raw_ostream &os) {
  SQLTranslator translator(os);
  auto moduleOp = llvm::dyn_cast<mlir::ModuleOp>(op);
  if (!moduleOp) {
    return op->emitOpError("expected a module");
  }

  return translator.translate(moduleOp);
}

} // namespace garel
