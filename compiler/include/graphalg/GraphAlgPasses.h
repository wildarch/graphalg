#pragma once

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Builders.h>
#include <mlir/Pass/Pass.h>

#include <graphalg/GraphAlgDialect.h>
#include <graphalg/GraphAlgOps.h>

namespace graphalg {

struct CallArgumentDimensions {
  std::uint64_t rows = 1;
  std::uint64_t cols = 1;
};

} // namespace graphalg

namespace llvm::cl {

template <>
class parser<graphalg::CallArgumentDimensions>
    : public basic_parser<graphalg::CallArgumentDimensions> {
public:
  parser(Option &o) : basic_parser(o) {}

  bool parse(Option &o, StringRef argName, StringRef arg,
             graphalg::CallArgumentDimensions &out);

  static void print(llvm::raw_ostream &os,
                    const graphalg::CallArgumentDimensions &value);
};

} // namespace llvm::cl

namespace graphalg {

/**
 * Builds the scalar op applying \c op to \c lhs and \c rhs.
 *
 * For example, if \c op is \c ADD, an \c AddOp is created.
 *
 * @return The result of applying \c op to \c lhs and \c rhs.
 */
mlir::FailureOr<mlir::Value> createScalarOpFor(mlir::Location loc, BinaryOp op,
                                               mlir::Value lhs, mlir::Value rhs,
                                               mlir::OpBuilder &builder);

#define GEN_PASS_DECL
#include "graphalg/GraphAlgPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "graphalg/GraphAlgPasses.h.inc"

// Testing only:
void registerTestDensePass();

} // namespace graphalg
