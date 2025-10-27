#include <optional>
#include <utility>
#include <vector>

#include <llvm/ADT/APFloat.h>
#include <llvm/ADT/APInt.h>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/DenseSet.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/StringSaver.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributeInterfaces.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/Types.h>
#include <mlir/Support/LLVM.h>

#include "graphalg/GraphAlgAttr.h"
#include "graphalg/GraphAlgCast.h"
#include "graphalg/GraphAlgDialect.h"
#include "graphalg/GraphAlgOps.h"
#include "graphalg/GraphAlgTypes.h"
#include "graphalg/SemiringTypes.h"
#include "graphalg/parse/Lexer.h"
#include "graphalg/parse/Parser.h"

namespace graphalg {

namespace {

/** Maps dimension names in the program text to dimension symbols. */
class DimMapper {
private:
  mlir::MLIRContext *_ctx;
  llvm::DenseMap<llvm::StringRef, DimAttr> _nameToDim;
  llvm::DenseMap<DimAttr, llvm::StringRef> _dimToName;

public:
  DimMapper(mlir::MLIRContext *ctx);
  DimAttr getOrAllocate(llvm::StringRef s);
  llvm::StringRef getName(DimAttr dim) const;
};

class TypeFormatter {
private:
  const DimMapper &_dimMapper;

  std::string _type;

  void formatScalar(mlir::Type t);
  void formatColumnVector(MatrixType t);
  void formatMatrix(MatrixType t);

public:
  TypeFormatter(const DimMapper &dimMapper) : _dimMapper(dimMapper) {}

  void format(mlir::Type t);

  std::string take() { return std::move(_type); }
};

class Parser {
private:
  llvm::ArrayRef<Token> _tokens;
  std::size_t _offset = 0;

  mlir::ModuleOp _module;
  mlir::OpBuilder _builder;

  // Value and location where the assignment happened
  struct VariableAssignment {
    mlir::Value value;
    std::optional<mlir::Location> loc;
  };
  llvm::ScopedHashTable<llvm::StringRef, VariableAssignment> _symbolTable;
  using VariableScope =
      llvm::ScopedHashTableScope<llvm::StringRef, VariableAssignment>;

  DimMapper _dimMapper;

  Token cur() { return _tokens[_offset]; }

  void eat() {
    _offset++;
    // NOTE: We should never eat the end of file token
    assert(_offset < _tokens.size() && "No next token");
  }

  mlir::ParseResult eatOrError(Token::Kind kind) {
    if (cur().type == kind) {
      eat();
      return mlir::success();
    } else {
      return mlir::emitError(cur().loc) << "expected " << Token::kindName(kind);
    }
  }

  std::string typeToString(mlir::Type t);
  std::string dimsToString(std::pair<DimAttr, DimAttr> dims);

  mlir::LogicalResult assign(mlir::Location loc, llvm::StringRef name,
                             mlir::Value value);
  bool isVarDefined(llvm::StringRef name) { return _symbolTable.count(name); }

  DimAttr inferDim(mlir::Value v, mlir::Location refLoc);

  mlir::ParseResult parseIdent(llvm::StringRef &s);
  mlir::ParseResult parseFuncRef(mlir::func::FuncOp &funcOp);
  mlir::ParseResult parseType(mlir::Type &t);
  mlir::ParseResult parseDim(DimAttr &t);

  mlir::Type tryParseSemiring();
  mlir::ParseResult parseSemiring(mlir::Type &s);

  mlir::ParseResult parseProgram();
  mlir::ParseResult parseFunction();
  mlir::ParseResult parseParams(llvm::SmallVectorImpl<llvm::StringRef> &names,
                                llvm::SmallVectorImpl<mlir::Type> &types,
                                llvm::SmallVectorImpl<mlir::Location> &locs);

  mlir::ParseResult parseBlock();
  mlir::ParseResult parseStmt();
  mlir::ParseResult parseStmtFor();
  mlir::ParseResult parseStmtReturn();
  mlir::ParseResult parseStmtAssign();
  mlir::ParseResult parseStmtAccum(mlir::Location baseLoc,
                                   llvm::StringRef baseName);

  struct ParsedMask {
    mlir::Value mask = nullptr;
    ;
    bool complement = false;

    bool isNone() const { return !mask; }
  };
  mlir::ParseResult parseMask(ParsedMask &mask);
  mlir::Value applyMask(mlir::Location baseLoc, mlir::Value base,
                        mlir::Location maskLoc, const ParsedMask &mask,
                        mlir::Location exprLoc, mlir::Value expr);

  enum class ParsedFill {
    /** A = v */
    NONE,
    /** A[:] = v */
    VECTOR,
    /** A[:, :] = v */
    MATRIX,
  };
  mlir::ParseResult parseFill(ParsedFill &fill);
  mlir::Value applyFill(mlir::Location baseLoc, mlir::Value base,
                        mlir::Location fillLoc, ParsedFill fill,
                        mlir::Location exprLoc, mlir::Value expr);

  struct ParsedRange {
    mlir::Value begin;
    mlir::Value end;
    DimAttr dim;
  };
  mlir::ParseResult parseRange(ParsedRange &r);

  mlir::ParseResult parseExpr(mlir::Value &v, int minPrec = 1);

  mlir::ParseResult parseBinaryOp(BinaryOp &op);

  mlir::Value buildMatMul(mlir::Location loc, mlir::Value lhs, mlir::Value rhs);

  mlir::ParseResult parseAtom(mlir::Value &v);

  mlir::ParseResult parseLiteral(mlir::Type ring, mlir::Value &v);

public:
  Parser(llvm::ArrayRef<Token> tokens, mlir::ModuleOp module)
      : _tokens(tokens), _module(module), _builder(module.getContext())
        /* , _stringPool(_stringAllocator)  */
        ,
        _dimMapper(module.getContext()) {}

  mlir::LogicalResult parse();
};

} // namespace

DimMapper::DimMapper(mlir::MLIRContext *ctx) : _ctx(ctx) {
  // The dialect must be loaded before we can use dimension symbols
  ctx->getOrLoadDialect<GraphAlgDialect>();

  // Mapping for the special '1' dimension
  auto oneDim = DimAttr::getOne(ctx);
  _nameToDim["1"] = oneDim;
  _dimToName[oneDim] = "1";
}

DimAttr DimMapper::getOrAllocate(llvm::StringRef name) {
  auto it = _nameToDim.find(name);
  if (it != _nameToDim.end()) {
    // Already defined
    return it->second;
  }

  // New dim
  auto dim = DimAttr::newAbstract(_ctx);
  _nameToDim[name] = dim;
  _dimToName[dim] = name;
  return dim;
}

llvm::StringRef DimMapper::getName(DimAttr dim) const {
  assert(_dimToName.contains(dim));
  return _dimToName.at(dim);
}

void TypeFormatter::formatScalar(mlir::Type t) {
  if (t.isInteger(/*width=*/1)) {
    _type += "bool";
  } else if (t.isInteger(/*width=*/64)) {
    _type += "int";
  } else if (t.isF64()) {
    _type += "real";
  } else if (llvm::isa<DimType>(t)) {
    _type += "dim";
  } else if (llvm::isa<TropI64Type>(t)) {
    _type += "trop_int";
  } else if (llvm::isa<TropF64Type>(t)) {
    _type += "trop_real";
  } else if (llvm::isa<TropMaxI64Type>(t)) {
    _type += "trop_max_int";
  } else {
    _type += "!!! UNKNOWN TYPE (";
    _type += t.getAbstractType().getName();
    _type += ") !!!";
  }
}

void TypeFormatter::formatColumnVector(MatrixType t) {
  assert(t.isColumnVector());
  _type += "Vector<";

  // Rows
  _type += _dimMapper.getName(t.getRows());
  _type += ", ";

  formatScalar(t.getSemiring());
  _type += ">";
}

void TypeFormatter::formatMatrix(MatrixType t) {
  if (t.isColumnVector()) {
    return formatColumnVector(t);
  }

  _type += "Matrix<";

  // Rows
  _type += _dimMapper.getName(t.getRows());
  _type += ", ";

  // Columns
  _type += _dimMapper.getName(t.getCols());
  _type += ", ";

  formatScalar(t.getSemiring());
  _type += ">";
}

void TypeFormatter::format(mlir::Type t) {
  if (auto mat = llvm::dyn_cast<MatrixType>(t)) {
    formatMatrix(mat);
  } else {
    formatScalar(t);
  }
}

std::string Parser::typeToString(mlir::Type type) {
  TypeFormatter fmt(_dimMapper);
  fmt.format(type);
  return fmt.take();
}

std::string Parser::dimsToString(std::pair<DimAttr, DimAttr> dims) {
  auto [r, c] = dims;
  return "(" + _dimMapper.getName(r).str() + " x " +
         _dimMapper.getName(c).str() + ")";
}

mlir::LogicalResult Parser::assign(mlir::Location loc, llvm::StringRef name,
                                   mlir::Value value) {
  auto previous = _symbolTable.lookup(name);
  if (previous.value && previous.value.getType() != value.getType()) {
    auto diag = mlir::emitError(loc)
                << "cannot assign value of type "
                << typeToString(value.getType())
                << " to previously defined variable of type "
                << typeToString(previous.value.getType());
    diag.attachNote(previous.loc) << "previous assigment was here";
    return diag;
  }

  _symbolTable.insert(name, {value, loc});
  return mlir::success();
}

DimAttr Parser::inferDim(mlir::Value v, mlir::Location refLoc) {
  auto dimOp = v.getDefiningOp<CastDimOp>();
  if (!dimOp) {
    auto diag = mlir::emitError(refLoc) << "not a dimension type";
    diag.attachNote(v.getLoc()) << "defined here";
    return nullptr;
  }

  return dimOp.getInput();
}

mlir::ParseResult Parser::parseIdent(llvm::StringRef &s) {
  if (cur().type != Token::IDENT) {
    return mlir::emitError(cur().loc) << "expected identifier";
  }

  s = cur().body;
  return eatOrError(Token::IDENT);
}

mlir::ParseResult Parser::parseFuncRef(mlir::func::FuncOp &funcOp) {
  auto loc = cur().loc;
  llvm::StringRef name;
  if (parseIdent(name)) {
    return mlir::failure();
  }

  funcOp =
      llvm::dyn_cast_if_present<mlir::func::FuncOp>(_module.lookupSymbol(name));
  if (!funcOp) {
    return mlir::emitError(loc) << "unknown function '" << name << "'";
  }

  return mlir::success();
}

mlir::ParseResult Parser::parseType(mlir::Type &t) {
  auto *ctx = _builder.getContext();
  if (auto ring = tryParseSemiring()) {
    t = MatrixType::scalarOf(ring);
    return mlir::success();
  } else if (cur().type == Token::IDENT && cur().body == "Matrix") {
    // Matrix
    DimAttr rows;
    DimAttr cols;
    mlir::Type ring;
    if (eatOrError(Token::IDENT) || eatOrError(Token::LANGLE) ||
        parseDim(rows) || eatOrError(Token::COMMA) || parseDim(cols) ||
        eatOrError(Token::COMMA) || parseSemiring(ring) ||
        eatOrError(Token::RANGLE)) {
      return mlir::failure();
    }

    t = MatrixType::get(ctx, rows, cols, ring);
    return mlir::success();
  } else if (cur().type == Token::IDENT && cur().body == "Vector") {
    DimAttr rows;
    mlir::Type ring;
    if (eatOrError(Token::IDENT) || eatOrError(Token::LANGLE) ||
        parseDim(rows) || eatOrError(Token::COMMA) || parseSemiring(ring) ||
        eatOrError(Token::RANGLE)) {
      return mlir::failure();
    }

    t = MatrixType::get(ctx, rows, DimAttr::getOne(ctx), ring);
    return mlir::success();
  }

  return mlir::emitError(cur().loc)
         << "expected type such as 'int', 'Matrix<..>' or 'Vector<..>'";
}

mlir::ParseResult Parser::parseDim(DimAttr &t) {
  if (cur().type == Token::INT && cur().body == "1") {
    t = DimAttr::getOne(_builder.getContext());
    return eatOrError(Token::INT);
  } else if (cur().type == Token::IDENT) {
    t = _dimMapper.getOrAllocate(cur().body);
    return eatOrError(Token::IDENT);
  }

  return mlir::emitError(cur().loc)
         << "expected a dimension symbol such as 's' or '1'";
}

mlir::Type Parser::tryParseSemiring() {
  if (cur().type != Token::IDENT) {
    return nullptr;
  }

  auto name = cur().body;
  auto *ctx = _builder.getContext();
  auto ring = llvm::StringSwitch<mlir::Type>(name)
                  .Case("bool", SemiringTypes::forBool(ctx))
                  .Case("int", SemiringTypes::forInt(ctx))
                  .Case("real", SemiringTypes::forReal(ctx))
                  .Case("trop_int", SemiringTypes::forTropInt(ctx))
                  .Case("trop_real", SemiringTypes::forTropReal(ctx))
                  .Case("trop_max_int", SemiringTypes::forTropMaxInt(ctx))
                  .Default(nullptr);
  if (ring) {
    (void)eatOrError(Token::IDENT);
  }

  return ring;
}

mlir::ParseResult Parser::parseSemiring(mlir::Type &s) {
  s = tryParseSemiring();
  return mlir::success(!!s);
}

mlir::ParseResult Parser::parseProgram() {
  _builder.setInsertionPointToStart(_module.getBody());
  while (cur().type == Token::FUNC) {
    if (mlir::failed(parseFunction())) {
      return mlir::failure();
    }
  }

  if (cur().type != Token::END_OF_FILE) {
    auto diag = mlir::emitError(cur().loc)
                << "invalid top-level definition, expected keyword 'func'";
    diag.attachNote() << "only function definitions are allowed here";
    return diag;
  }

  return mlir::success();
}

mlir::ParseResult Parser::parseFunction() {
  llvm::StringRef name;
  llvm::SmallVector<llvm::StringRef> paramNames;
  llvm::SmallVector<mlir::Type> paramTypes;
  llvm::SmallVector<mlir::Location> paramLocs;
  mlir::Type returnType;
  auto loc = cur().loc;
  if (eatOrError(Token::FUNC) || parseIdent(name) ||
      parseParams(paramNames, paramTypes, paramLocs) ||
      eatOrError(Token::ARROW) || parseType(returnType)) {
    return mlir::failure();
  }

  // TODO: Check duplicate definition

  // Create the new op.
  auto funcType = _builder.getFunctionType(paramTypes, {returnType});
  auto funcOp = _builder.create<mlir::func::FuncOp>(loc, name, funcType);

  // Populate the function body.
  mlir::OpBuilder::InsertionGuard guard(_builder);
  auto &entryBlock = funcOp.getFunctionBody().emplaceBlock();
  _builder.setInsertionPointToStart(&entryBlock);
  VariableScope functionScope(_symbolTable);
  for (const auto &[name, type, loc] :
       llvm::zip_equal(paramNames, paramTypes, paramLocs)) {
    auto arg = entryBlock.addArgument(type, loc);
    if (mlir::failed(assign(loc, name, arg))) {
      return mlir::failure();
    }
  }

  if (mlir::failed(parseBlock())) {
    return mlir::failure();
  }

  // TODO: Check for return statement.

  return mlir::success();
}

mlir::ParseResult
Parser::parseParams(llvm::SmallVectorImpl<llvm::StringRef> &names,
                    llvm::SmallVectorImpl<mlir::Type> &types,
                    llvm::SmallVectorImpl<mlir::Location> &locs) {
  if (eatOrError(Token::LPAREN)) {
    return mlir::failure();
  }

  if (cur().type == Token::RPAREN) {
    // No parameters
    return eatOrError(Token::RPAREN);
  }

  // First parameter
  auto &name = names.emplace_back();
  auto &type = types.emplace_back();
  locs.emplace_back(cur().loc);
  if (parseIdent(name) || eatOrError(Token::COLON) || parseType(type)) {
    return mlir::failure();
  }

  while (cur().type != Token::RPAREN) {
    // More parameters
    if (eatOrError(Token::COMMA)) {
      return mlir::failure();
    }

    auto &name = names.emplace_back();
    auto &type = types.emplace_back();
    locs.emplace_back(cur().loc);
    if (parseIdent(name) || eatOrError(Token::COLON) || parseType(type)) {
      return mlir::failure();
    }
  }

  return eatOrError(Token::RPAREN);
}

mlir::ParseResult Parser::parseBlock() {
  if (eatOrError(Token::LBRACE)) {
    return mlir::failure();
  }

  while (cur().type != Token::RBRACE && cur().type != Token::END_OF_FILE) {
    if (parseStmt()) {
      return mlir::failure();
    }
  }

  if (eatOrError(Token::RBRACE)) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::ParseResult Parser::parseStmt() {
  switch (cur().type) {
  case Token::FOR:
    return parseStmtFor();
  case Token::RETURN:
    return parseStmtReturn();
  case Token::IDENT:
    return parseStmtAssign();
  default:
    return mlir::emitError(cur().loc) << "invalid start for statement";
  }
}

static void
findModifiedBindingsInBlock(llvm::ArrayRef<Token> tokens, std::size_t offset,
                            llvm::SmallVectorImpl<llvm::StringRef> &bindings) {
  llvm::SmallDenseSet<llvm::StringRef> uniqueBindings;
  if (tokens[offset].type != Token::LBRACE) {
    // Block must start with {
    return;
  }

  offset++;
  std::size_t depth = 1;
  for (; offset + 1 < tokens.size() && depth > 0; offset++) {
    auto cur = tokens[offset];
    switch (cur.type) {
    case Token::IDENT: {
      auto name = cur.body;
      auto next = tokens[offset + 1];
      if (next.type == Token::ASSIGN || next.type == Token::ACCUM) {
        // An assignment or accumulate op
        auto [_, newlyAdded] = uniqueBindings.insert(name);
        if (newlyAdded) {
          bindings.emplace_back(name);
        }
      }
    }
    case Token::LBRACE: {
      // Enter nested block.
      depth++;
      break;
    }
    case Token::RBRACE: {
      // End of block.
      depth--;
      break;
    }
    default:
      // Skip
      break;
    }
  }
}

mlir::ParseResult Parser::parseStmtFor() {
  auto loc = cur().loc;
  llvm::StringRef iterVarName;
  ParsedRange range;
  if (eatOrError(Token::FOR) || parseIdent(iterVarName) ||
      eatOrError(Token::IN) || parseRange(range)) {
    return mlir::failure();
  }

  // Find the variables modified inside the loop.
  llvm::SmallVector<llvm::StringRef> varNames;
  findModifiedBindingsInBlock(_tokens, _offset, varNames);
  // Only the variables that exist outside the loop are proper loop variables.
  auto removeIt = llvm::remove_if(
      varNames, [&](llvm::StringRef name) { return !isVarDefined(name); });
  varNames.erase(removeIt, varNames.end());
  llvm::SmallVector<mlir::Value> initArgs;
  llvm::SmallVector<mlir::Type> varTypes;
  for (auto name : varNames) {
    auto [value, _] = _symbolTable.lookup(name);
    assert(!!value);

    initArgs.emplace_back(value);
    varTypes.emplace_back(value.getType());
  }

  // Create the for op.
  mlir::Region *bodyRegion;
  mlir::Region *untilRegion;
  mlir::ValueRange results;
  if (range.dim) {
    auto forOp = _builder.create<ForDimOp>(loc, varTypes, initArgs, range.dim);
    bodyRegion = &forOp.getBody();
    untilRegion = &forOp.getUntil();
    results = forOp->getResults();
  } else {
    assert(range.begin && range.end);
    auto forOp = _builder.create<ForConstOp>(loc, varTypes, initArgs,
                                             range.begin, range.end);
    bodyRegion = &forOp.getBody();
    untilRegion = &forOp.getUntil();
    results = forOp->getResults();
  }

  // Create body.
  assert(bodyRegion->getBlocks().empty());
  auto &bodyBlock = bodyRegion->emplaceBlock();

  {
    // Builder and variable scope for this block
    VariableScope bodyScope(_symbolTable);
    mlir::OpBuilder::InsertionGuard guard(_builder);
    _builder.setInsertionPointToStart(&bodyBlock);
    // Define the iteration variable
    auto iterVar = bodyBlock.addArgument(
        MatrixType::scalarOf(SemiringTypes::forInt(_builder.getContext())),
        loc);
    if (mlir::failed(assign(loc, iterVarName, iterVar))) {
      return mlir::failure();
    }

    // Map variables to their values in the body block
    for (const auto &[name, type] : llvm::zip_equal(varNames, varTypes)) {
      auto var = bodyBlock.addArgument(type, loc);
      if (mlir::failed(assign(loc, name, var))) {
        return mlir::failure();
      }
    }

    // Parse body block.
    if (mlir::failed(parseBlock())) {
      return mlir::failure();
    }

    // yield the new values for the variables
    llvm::SmallVector<mlir::Value> yieldInputs;
    for (const auto &name : varNames) {
      auto [newValue, _] = _symbolTable.lookup(name);
      assert(!!newValue);
      yieldInputs.emplace_back(newValue);
    }
    _builder.create<YieldOp>(loc, yieldInputs);
  }

  if (cur().type == Token::UNTIL) {
    auto loc = cur().loc;
    eat();

    // Add until block
    assert(untilRegion->getBlocks().empty());
    auto &untilBlock = untilRegion->emplaceBlock();

    // Builder and variable scope for this block
    VariableScope bodyScope(_symbolTable);
    mlir::OpBuilder::InsertionGuard guard(_builder);
    _builder.setInsertionPointToStart(&untilBlock);
    // Define the iteration variable
    auto iterVar = untilBlock.addArgument(
        MatrixType::scalarOf(_builder.getI64Type()), loc);
    if (mlir::failed(assign(loc, iterVarName, iterVar))) {
      return mlir::failure();
    }

    // Map variables to their values in the until block
    for (const auto &[name, type] : llvm::zip_equal(varNames, varTypes)) {
      auto var = untilBlock.addArgument(type, loc);
      if (mlir::failed(assign(loc, name, var))) {
        return mlir::failure();
      }
    }

    // Parse condtion expression.
    mlir::Value result;
    loc = cur().loc;
    if (parseExpr(result) || eatOrError(Token::SEMI)) {
      return mlir::failure();
    }

    auto resultType = llvm::dyn_cast<MatrixType>(result.getType());
    if (!resultType || !resultType.isScalar() || !resultType.isBoolean()) {
      return mlir::emitError(loc)
             << "loop condition does not produce a boolean scalar, got "
             << typeToString(resultType);
    }

    _builder.create<YieldOp>(loc, result);
  }

  // Use the updated variables from the block
  assert(results.size() == varNames.size());
  assert(results.getTypes() == varTypes);
  for (auto [name, value] : llvm::zip_equal(varNames, results)) {
    if (mlir::failed(assign(loc, name, value))) {
      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::ParseResult Parser::parseStmtReturn() {
  auto loc = cur().loc;
  mlir::Value returnValue;
  if (eatOrError(Token::RETURN) || parseExpr(returnValue) ||
      eatOrError(Token::SEMI)) {
    return mlir::failure();
  }

  _builder.create<mlir::func::ReturnOp>(loc, returnValue);
  return mlir::success();
}

mlir::ParseResult Parser::parseStmtAssign() {
  auto loc = cur().loc;
  llvm::StringRef baseName;
  if (parseIdent(baseName)) {
    return mlir::failure();
  }

  if (cur().type == Token::ACCUM) {
    return parseStmtAccum(loc, baseName);
  }

  ParsedMask mask;
  auto maskLoc = cur().loc;
  if (cur().type == Token::LANGLE) {
    if (mlir::failed(parseMask(mask))) {
      return mlir::failure();
    }
  }

  auto fill = ParsedFill::NONE;
  auto fillLoc = cur().loc;
  if (cur().type == Token::LSBRACKET) {
    if (mlir::failed(parseFill(fill))) {
      return mlir::failure();
    }
  }

  if (eatOrError(Token::ASSIGN)) {
    return mlir::failure();
  }

  auto exprLoc = cur().loc;
  mlir::Value expr;
  if (parseExpr(expr) || eatOrError(Token::SEMI)) {
    return mlir::failure();
  }

  auto [baseValue, _] = _symbolTable.lookup(baseName);
  if (!baseValue) {
    // New variable
    if (fill != ParsedFill::NONE || !mask.isNone()) {
      // Cannot fill or mask if base was not already defined.
      return mlir::emitError(loc) << "undefined variable '" << baseName << "'";
    }

    // Regular assignment.
    return assign(loc, baseName, expr);
  }

  expr = applyFill(loc, baseValue, fillLoc, fill, exprLoc, expr);
  if (!expr) {
    return mlir::failure();
  }

  expr = applyMask(loc, baseValue, maskLoc, mask, exprLoc, expr);
  if (!expr) {
    return mlir::failure();
  }

  return assign(loc, baseName, expr);
}

mlir::ParseResult Parser::parseMask(ParsedMask &mask) {
  if (eatOrError(Token::LANGLE)) {
    return mlir::failure();
  }

  if (cur().type == Token::NOT) {
    eat();
    mask.complement = true;
  }

  auto loc = cur().loc;
  llvm::StringRef name;
  if (parseIdent(name)) {
    return mlir::failure();
  }

  auto [maskValue, _] = _symbolTable.lookup(name);
  if (!maskValue) {
    return mlir::emitError(loc) << "undefined variable '" << name << "'";
  }

  mask.mask = maskValue;
  return eatOrError(Token::RANGLE);
}

mlir::Value Parser::applyMask(mlir::Location baseLoc, mlir::Value base,
                              mlir::Location maskLoc, const ParsedMask &mask,
                              mlir::Location exprLoc, mlir::Value expr) {
  if (mask.isNone()) {
    // no mask to apply
    return expr;
  }

  auto baseType = llvm::cast<MatrixType>(base.getType());
  auto maskType = llvm::cast<MatrixType>(mask.mask.getType());
  auto exprType = llvm::cast<MatrixType>(expr.getType());

  if (baseType != exprType) {
    auto diag = mlir::emitError(baseLoc)
                << "base type does not match the value to assign";
    diag.attachNote(baseLoc) << "base type: " << typeToString(baseType);
    diag.attachNote(exprLoc) << "expression type: " << typeToString(exprType);
    return nullptr;
  } else if (!maskType.isBoolean()) {
    // TODO: Widen this to allow any semiring (using implicit cast)
    mlir::emitError(maskLoc)
        << "mask is not a boolean matrix: " << typeToString(maskType);
    return nullptr;
  }

  auto baseDims = baseType.getDims();
  auto maskDims = maskType.getDims();
  if (baseDims != maskDims) {
    auto diag = mlir::emitError(baseLoc)
                << "base dimensions do not match the dimensions of the mask";
    diag.attachNote(baseLoc) << "base dimension: " << dimsToString(baseDims);
    diag.attachNote(maskLoc) << "mask dimensions: " << dimsToString(maskDims);
    return nullptr;
  }

  return _builder.create<MaskOp>(maskLoc, base, mask.mask, expr,
                                 mask.complement);
}

mlir::ParseResult Parser::parseFill(ParsedFill &fill) {
  // [:
  if (eatOrError(Token::LSBRACKET) || eatOrError(Token::COLON)) {
    return mlir::failure();
  }

  if (cur().type == Token::COMMA) {
    fill = ParsedFill::MATRIX;

    // ,:]
    if (eatOrError(Token::COMMA) || eatOrError(Token::COLON) ||
        eatOrError(Token::RSBRACKET)) {
      return mlir::failure();
    }

    return mlir::success();
  } else {
    fill = ParsedFill::VECTOR;
    return eatOrError(Token::RSBRACKET);
  }
}

mlir::Value Parser::applyFill(mlir::Location baseLoc, mlir::Value base,
                              mlir::Location fillLoc, ParsedFill fill,
                              mlir::Location exprLoc, mlir::Value expr) {
  if (fill == ParsedFill::NONE) {
    // No fill to apply
    return expr;
  }

  auto baseType = llvm::cast<MatrixType>(base.getType());
  auto exprType = llvm::cast<MatrixType>(expr.getType());
  if (!exprType.isScalar()) {
    auto diag = mlir::emitError(exprLoc) << "fill expression is not a scalar";
    return nullptr;
  }

  auto baseRing = baseType.getSemiring();
  auto exprRing = exprType.getSemiring();
  if (baseRing != exprRing) {
    auto diag = mlir::emitError(baseLoc)
                << "base and fill expression have different semirings";
    diag.attachNote(exprLoc)
        << "fill expression has semiring " << typeToString(exprRing);
    diag.attachNote(baseLoc)
        << "base matrix has semiring " << typeToString(baseRing);
    return nullptr;
  }

  if (fill == ParsedFill::VECTOR && !baseType.isColumnVector()) {
    auto diag = mlir::emitError(fillLoc)
                << "vector fill [:] used with non-vector base";
    diag.attachNote(baseLoc) << "base has type " << typeToString(baseType);
    return nullptr;
  }

  return _builder.create<BroadcastOp>(fillLoc, baseType, expr);
}

mlir::ParseResult Parser::parseStmtAccum(mlir::Location baseLoc,
                                         llvm::StringRef baseName) {
  mlir::Value expr;
  if (eatOrError(Token::ACCUM) || parseExpr(expr) || eatOrError(Token::SEMI)) {
    return mlir::failure();
  }

  auto [baseValue, _] = _symbolTable.lookup(baseName);
  if (!baseValue) {
    return mlir::emitError(baseLoc) << "undefined variable";
  } else if (baseValue.getType() != expr.getType()) {
    return mlir::emitError(baseLoc)
           << "type of base does not match the expression to accumulate: ("
           << typeToString(baseValue.getType()) << " vs. "
           << typeToString(expr.getType());
  }

  // Rewrite a += b; to a = a (.+) b;
  auto result = mlir::Value(
      _builder.create<ElementWiseOp>(baseLoc, baseValue, BinaryOp::ADD, expr));
  return assign(baseLoc, baseName, result);
}

mlir::ParseResult Parser::parseRange(ParsedRange &r) {
  auto exprLoc = cur().loc;
  mlir::Value expr;
  if (parseExpr(expr)) {
    return mlir::failure();
  }

  if (cur().type == Token::COLON) {
    // Const range
    r.begin = expr;
    if (eatOrError(Token::COLON) || parseExpr(r.end)) {
      return mlir::failure();
    }

    return mlir::success();
  } else {
    r.dim = inferDim(expr, exprLoc);
    if (!r.dim) {
      return mlir::failure();
    }

    return mlir::success();
  }
}

static int precedenceForOp(Token::Kind op) {
  switch (op) {
  // NOTE: 1 for ewise
  case Token::EQUAL:
  case Token::NOT_EQUAL:
  case Token::LANGLE:
  case Token::RANGLE:
  case Token::LEQ:
  case Token::GEQ:
    return 2;
  case Token::PLUS:
  case Token::MINUS:
    return 3;
  case Token::TIMES:
  case Token::DIVIDE:
    return 4;
  case Token::DOT:
    return 5;
  default:
    // Not an op with precedence
    return 0;
  }
}

mlir::ParseResult Parser::parseExpr(mlir::Value &v, int minPrec) {
  mlir::Value atomLhs;
  if (parseAtom(atomLhs)) {
    return mlir::failure();
  }

  while (true) {
    bool ewise = cur().type == Token::LPAREN && _offset + 1 < _tokens.size() &&
                 _tokens[_offset + 1].type == Token::DOT;
    int prec = ewise ? 1 : precedenceForOp(cur().type);
    if (!prec || prec < minPrec) {
      break;
    }

    int nextMinPrec = prec + 1;
    if (cur().type == Token::DOT) {
      // Matrix property
      eat();
      if (cur().type != Token::IDENT) {
        return mlir::emitError(cur().loc)
               << "expected matrix property such as 'nrows'";
      }

      if (cur().body == "T") {
        atomLhs = _builder.create<TransposeOp>(cur().loc, atomLhs);
      } else if (cur().body == "nrows") {
        auto matType = llvm::cast<MatrixType>(atomLhs.getType());
        atomLhs = _builder.create<CastDimOp>(cur().loc, matType.getRows());
      } else if (cur().body == "ncols") {
        auto matType = llvm::cast<MatrixType>(atomLhs.getType());
        atomLhs = _builder.create<CastDimOp>(cur().loc, matType.getCols());
      } else if (cur().body == "nvals") {
        atomLhs = _builder.create<NValsOp>(cur().loc, atomLhs);
      } else {
        return mlir::emitError(cur().loc) << "invalid matrix property";
      }

      eat();
    } else if (ewise) {
      eat(); // '('
      eat(); // '.'
      if (cur().type == Token::IDENT) {
        // element-wise function
        auto loc = cur().loc;
        mlir::func::FuncOp funcOp;
        mlir::Value atomRhs;
        if (parseFuncRef(funcOp) || eatOrError(Token::RPAREN) ||
            parseExpr(atomRhs, nextMinPrec)) {
          return mlir::failure();
        }

        atomLhs =
            _builder.create<ApplyElementWiseOp>(loc, funcOp, atomLhs, atomRhs);
      } else {
        // element-wise binop
        auto loc = cur().loc;
        BinaryOp binop;
        mlir::Value atomRhs;
        if (parseBinaryOp(binop) || eatOrError(Token::RPAREN) ||
            parseExpr(atomRhs, nextMinPrec)) {
          return mlir::failure();
        }

        atomLhs = _builder.create<ElementWiseOp>(loc, atomLhs, binop, atomRhs);
      }
    } else {
      // Binary operator
      auto loc = cur().loc;
      BinaryOp binop;
      mlir::Value atomRhs;
      if (parseBinaryOp(binop) || parseExpr(atomRhs, nextMinPrec)) {
        return mlir::failure();
      }

      if (binop == BinaryOp::MUL) {
        // Matmul special case
        atomLhs = buildMatMul(loc, atomLhs, atomRhs);
      } else {
        // TODO: check scalar matrix types
        atomLhs = _builder.create<ElementWiseOp>(loc, atomLhs, binop, atomRhs);
      }
    }
  }

  v = atomLhs;
  return mlir::success();
}

mlir::ParseResult Parser::parseBinaryOp(BinaryOp &op) {
  switch (cur().type) {
  case Token::PLUS:
    op = BinaryOp::ADD;
    break;
  case Token::MINUS:
    op = BinaryOp::SUB;
    break;
  case Token::TIMES:
    op = BinaryOp::MUL;
    break;
  case Token::DIVIDE:
    op = BinaryOp::DIV;
    break;
  case Token::EQUAL:
    op = BinaryOp::EQ;
    break;
  case Token::NOT_EQUAL:
    op = BinaryOp::NE;
    break;
  case Token::LANGLE:
    op = BinaryOp::LT;
    break;
  case Token::RANGLE:
    op = BinaryOp::GT;
    break;
  case Token::LEQ:
    op = BinaryOp::LE;
    break;
  case Token::GEQ:
    op = BinaryOp::GE;
    break;
  default:
    return mlir::emitError(cur().loc) << "expected a binary operator";
  }

  eat();
  return mlir::success();
}

mlir::Value Parser::buildMatMul(mlir::Location loc, mlir::Value lhs,
                                mlir::Value rhs) {
  auto lhsType = llvm::cast<MatrixType>(lhs.getType());
  auto rhsType = llvm::cast<MatrixType>(rhs.getType());
  if (lhsType.getSemiring() != rhsType.getSemiring()) {
    auto diag = mlir::emitError(loc)
                << "incompatible semirings for matrix multiply";
    diag.attachNote(lhs.getLoc())
        << "left side has semiring " << typeToString(lhsType.getSemiring());
    diag.attachNote(rhs.getLoc())
        << "right side has semiring " << typeToString(rhsType.getSemiring());
    return nullptr;
  }

  if (lhsType.getCols() == rhsType.getRows()) {
    return _builder.create<MatMulOp>(loc, lhs, rhs);
  } else if (lhsType.isColumnVector() &&
             lhsType.getRows() == rhsType.getRows()) {
    // Special case: Allow implicit column to row vector transpose.
    return _builder.create<VecMatMulOp>(loc, lhs, rhs);
  }

  auto diag = mlir::emitError(loc)
              << "incompatible dimensions for matrix multiply";
  diag.attachNote(lhs.getLoc())
      << "left side has dimensions " << dimsToString(lhsType.getDims());
  diag.attachNote(rhs.getLoc())
      << "right side has dimensions " << dimsToString(rhsType.getDims());
  return nullptr;
}

mlir::ParseResult Parser::parseAtom(mlir::Value &v) {
  auto loc = cur().loc;
  switch (cur().type) {
  case Token::LPAREN: {
    // (<expr>)
    if (eatOrError(Token::LPAREN) || parseExpr(v) ||
        eatOrError(Token::RPAREN)) {
      return mlir::failure();
    }

    return mlir::success();
  }
  case Token::IDENT: {
    // <ring>(<literal>)
    if (auto ring = tryParseSemiring()) {
      // e.g. int(42)
      if (eatOrError(Token::LPAREN) || parseLiteral(ring, v) ||
          eatOrError(Token::RPAREN)) {
        return mlir::failure();
      }

      return mlir::success();
    }

    llvm::StringRef name;
    if (parseIdent(name)) {
      return mlir::failure();
    }

    if (name == "Matrix") {
      mlir::Type ring;
      mlir::Value rowsExpr;
      mlir::Value colsExpr;
      if (eatOrError(Token::LANGLE) || parseSemiring(ring) ||
          eatOrError(Token::RANGLE) || eatOrError(Token::LPAREN) ||
          parseExpr(rowsExpr) || eatOrError(Token::COMMA) ||
          parseExpr(colsExpr) || eatOrError(Token::RPAREN)) {
        return mlir::failure();
      }

      // TODO: Better ref locs
      auto rows = inferDim(rowsExpr, loc);
      auto cols = inferDim(colsExpr, loc);
      if (!rows || !cols) {
        return mlir::failure();
      }

      v = _builder.create<ConstantMatrixOp>(
          loc, _builder.getType<MatrixType>(rows, cols, ring),
          llvm::cast<SemiringTypeInterface>(ring).addIdentity());
      return mlir::success();
    }

    if (name == "Vector") {
      mlir::Type ring;
      mlir::Value rowsExpr;
      if (eatOrError(Token::LANGLE) || parseSemiring(ring) ||
          eatOrError(Token::RANGLE) || eatOrError(Token::LPAREN) ||
          parseExpr(rowsExpr) || eatOrError(Token::RPAREN)) {
        return mlir::failure();
      }

      // TODO: Better ref locs
      auto rows = inferDim(rowsExpr, loc);
      if (!rows) {
        return mlir::failure();
      }

      auto *ctx = _builder.getContext();
      v = _builder.create<ConstantMatrixOp>(
          loc, MatrixType::get(ctx, rows, DimAttr::getOne(ctx), ring),
          llvm::cast<SemiringTypeInterface>(ring).addIdentity());
      return mlir::success();
    }

    if (name == "cast") {
      mlir::Type ring;
      mlir::Value expr;
      if (eatOrError(Token::LANGLE) || parseSemiring(ring) ||
          eatOrError(Token::RANGLE) || eatOrError(Token::LPAREN) ||
          parseExpr(expr) || eatOrError(Token::RPAREN)) {
        return mlir::failure();
      }

      auto exprType = llvm::cast<MatrixType>(expr.getType());
      auto *dialect =
          _builder.getContext()->getLoadedDialect<GraphAlgDialect>();
      if (!dialect->isCastLegal(exprType.getSemiring(), ring)) {
        return mlir::emitError(loc)
               << "invalid cast from " << typeToString(exprType.getSemiring())
               << " to " << typeToString(ring);
      }

      v = _builder.create<CastOp>(
          loc,
          _builder.getType<MatrixType>(exprType.getRows(), exprType.getCols(),
                                       ring),
          expr);
      return mlir::success();
    }

    if (name == "zero") {
      mlir::Type ring;
      if (eatOrError(Token::LPAREN) || parseSemiring(ring) ||
          eatOrError(Token::RPAREN)) {
        return mlir::failure();
      }

      auto value = llvm::cast<SemiringTypeInterface>(ring).addIdentity();
      v = _builder.create<LiteralOp>(loc, value);
      return mlir::success();
    }

    if (name == "one") {
      mlir::Type ring;
      if (eatOrError(Token::LPAREN) || parseSemiring(ring) ||
          eatOrError(Token::RPAREN)) {
        return mlir::failure();
      }

      auto value = llvm::cast<SemiringTypeInterface>(ring).mulIdentity();
      v = _builder.create<LiteralOp>(loc, value);
      return mlir::success();
    }

    if (name == "apply") {
      mlir::func::FuncOp func;
      llvm::SmallVector<mlir::Value, 2> args(1);
      if (eatOrError(Token::LPAREN) || parseFuncRef(func) ||
          eatOrError(Token::COMMA) || parseExpr(args[0])) {
        return mlir::failure();
      }

      if (cur().type == Token::COMMA) {
        // Have a second arg.
        auto &arg = args.emplace_back();
        if (eatOrError(Token::COMMA) || parseExpr(arg)) {
          return mlir::failure();
        }
      }

      if (eatOrError(Token::RPAREN)) {
        return mlir::failure();
      }

      if (args.size() == 1) {
        v = _builder.create<ApplyUnaryOp>(loc, func, args[0]);
      } else {
        assert(args.size() == 2);
        v = _builder.create<ApplyBinaryOp>(loc, func, args[0], args[1]);
      }

      return mlir::success();
    }

    if (name == "select") {
      mlir::func::FuncOp func;
      llvm::SmallVector<mlir::Value, 2> args(1);
      if (eatOrError(Token::LPAREN) || parseFuncRef(func) ||
          eatOrError(Token::COMMA) || parseExpr(args[0])) {
        return mlir::failure();
      }

      if (cur().type == Token::COMMA) {
        // Have a second arg.
        auto &arg = args.emplace_back();
        if (eatOrError(Token::COMMA) || parseExpr(arg)) {
          return mlir::failure();
        }
      }

      if (eatOrError(Token::RPAREN)) {
        return mlir::failure();
      }

      if (args.size() == 1) {
        v = _builder.create<SelectUnaryOp>(loc, func.getSymName(), args[0]);
      } else {
        assert(args.size() == 2);
        v = _builder.create<SelectBinaryOp>(loc, func.getSymName(), args[0],
                                            args[1]);
      }

      return mlir::success();
    }

    if (name == "reduceRows") {
      mlir::Value arg;
      if (eatOrError(Token::LPAREN) || parseExpr(arg) ||
          eatOrError(Token::RPAREN)) {
        return mlir::failure();
      }

      auto inputType = llvm::cast<MatrixType>(arg.getType());
      auto *ctx = _builder.getContext();
      auto resultType =
          MatrixType::get(ctx, inputType.getRows(), DimAttr::getOne(ctx),
                          inputType.getSemiring());
      v = _builder.create<ReduceOp>(loc, resultType, arg);
      return mlir::success();
    }

    if (name == "reduceCols") {
      mlir::Value arg;
      if (eatOrError(Token::LPAREN) || parseExpr(arg) ||
          eatOrError(Token::RPAREN)) {
        return mlir::failure();
      }

      auto inputType = llvm::cast<MatrixType>(arg.getType());
      auto *ctx = _builder.getContext();
      auto resultType =
          MatrixType::get(ctx, DimAttr::getOne(ctx), inputType.getCols(),
                          inputType.getSemiring());
      v = _builder.create<ReduceOp>(loc, resultType, arg);
      return mlir::success();
    }

    if (name == "reduce") {
      mlir::Value arg;
      if (eatOrError(Token::LPAREN) || parseExpr(arg) ||
          eatOrError(Token::RPAREN)) {
        return mlir::failure();
      }

      auto inputType = llvm::cast<MatrixType>(arg.getType());
      v = _builder.create<ReduceOp>(loc, inputType.asScalar(), arg);
      return mlir::success();
    }

    if (name == "pickAny") {
      mlir::Value arg;
      if (eatOrError(Token::LPAREN) || parseExpr(arg) ||
          eatOrError(Token::RPAREN)) {
        return mlir::failure();
      }

      v = _builder.create<PickAnyOp>(loc, arg);
      return mlir::success();
    }

    if (name == "diag") {
      mlir::Value arg;
      if (eatOrError(Token::LPAREN) || parseExpr(arg) ||
          eatOrError(Token::RPAREN)) {
        return mlir::failure();
      }

      v = _builder.create<DiagOp>(loc, arg);
      return mlir::success();
    }

    // TODO: Make a separate extension
    if (name == "tril") {
      mlir::Value arg;
      if (eatOrError(Token::LPAREN) || parseExpr(arg) ||
          eatOrError(Token::RPAREN)) {
        return mlir::failure();
      }

      v = _builder.create<TrilOp>(loc, arg);
      return mlir::success();
    }

    // TODO: Make a separate extension
    if (name == "triu") {
      mlir::Value arg;
      if (eatOrError(Token::LPAREN) || parseExpr(arg) ||
          eatOrError(Token::RPAREN)) {
        return mlir::failure();
      }

      v = _builder.create<TriuOp>(loc, arg);
      return mlir::success();
    }

    auto var = _symbolTable.lookup(name);
    if (!var.value) {
      return mlir::emitError(loc) << "unrecognized variable";
    }

    v = var.value;
    return mlir::success();
  }
  case Token::NOT: {
    if (eatOrError(Token::NOT) || parseAtom(v)) {
      return mlir::failure();
    }

    v = _builder.create<NotOp>(loc, v);
    return mlir::success();
  }
  case Token::MINUS: {
    if (eatOrError(Token::MINUS) || parseAtom(v)) {
      return mlir::failure();
    }

    v = _builder.create<NegOp>(loc, v);
    return mlir::success();
  }
  default:
    return mlir::emitError(cur().loc) << "invalid expression";
  }
}

static std::optional<llvm::APInt> parseInt(llvm::StringRef s) {
  // The largest possible 64-bit signed integer has 19 characters.
  constexpr int maxCharacters = 19;
  assert(std::to_string(std::numeric_limits<std::int64_t>::max()).size() ==
         maxCharacters);
  if (s.size() <= maxCharacters) {
    // 128 bits is enough for any 19 character radix 10 integer.
    llvm::APInt v(128, s, 10);
    if (v.getSignificantBits() <= 64) {
      // Fits in 64 bits.
      return v.trunc(64);
    }
  }

  return std::nullopt;
}

static std::optional<llvm::APFloat> parseFloat(llvm::StringRef s) {
  llvm::APFloat v(llvm::APFloat::IEEEdouble());
  auto result = v.convertFromString(s, llvm::APFloat::rmNearestTiesToEven);
  // Note: decimal literals may not be representable in exact form in IEEE
  // double format.
  auto allowedStatusMask = llvm::APFloat::opOK | llvm::APFloat::opInexact;
  if (result.takeError() || (*result & (~allowedStatusMask)) != 0) {
    return std::nullopt;
  }

  return v;
}

mlir::ParseResult Parser::parseLiteral(mlir::Type ring, mlir::Value &v) {
  auto *ctx = _builder.getContext();
  mlir::TypedAttr attr;
  if (ring == SemiringTypes::forBool(ctx)) {
    if (cur().type == Token::FALSE) {
      attr = _builder.getBoolAttr(false);
    } else if (cur().type == Token::TRUE) {
      attr = _builder.getBoolAttr(true);
    } else {
      return mlir::emitError(cur().loc) << "expected 'true' or 'false'";
    }
  } else if (ring == SemiringTypes::forInt(ctx) ||
             ring == SemiringTypes::forTropInt(ctx) ||
             ring == SemiringTypes::forTropMaxInt(ctx)) {
    if (cur().type != Token::INT) {
      return mlir::emitError(cur().loc) << "expected an integer value";
    }

    if (auto v = parseInt(cur().body)) {
      auto intAttr = _builder.getIntegerAttr(SemiringTypes::forInt(ctx), *v);
      if (intAttr.getType() == ring) {
        attr = intAttr;
      } else {
        attr = TropIntAttr::get(ctx, ring, intAttr);
      }
    } else {
      return mlir::emitError(cur().loc) << "integer does not fit in 64 bits";
    }
  } else if (ring == SemiringTypes::forReal(ctx) ||
             ring == SemiringTypes::forTropReal(ctx)) {
    if (cur().type != Token::FLOAT) {
      return mlir::emitError(cur().loc) << "expected a floating-point value";
    }

    if (auto v = parseFloat(cur().body)) {
      auto floatAttr = _builder.getFloatAttr(SemiringTypes::forReal(ctx), *v);
      if (floatAttr.getType() == ring) {
        attr = floatAttr;
      } else {
        attr = TropFloatAttr::get(ctx, ring, floatAttr);
      }
    } else {
      return mlir::emitError(cur().loc) << "invalid floating-point literal";
    }
  } else {
    llvm_unreachable("Invalid semiring");
  }

  assert(!!attr);
  // Eat the literal token
  eat();
  v = _builder.create<LiteralOp>(cur().loc, MatrixType::scalarOf(ring), attr);
  return mlir::success();
}

mlir::LogicalResult Parser::parse() {
  auto *ctx = _builder.getContext();
  ctx->getOrLoadDialect<graphalg::GraphAlgDialect>();
  ctx->getOrLoadDialect<mlir::func::FuncDialect>();
  return parseProgram();
}

mlir::LogicalResult parse(llvm::StringRef program, mlir::ModuleOp moduleOp) {
  llvm::StringRef filename = "<unknown>";
  int startLine = 0;
  int startCol = 0;
  auto loc = llvm::dyn_cast<mlir::FileLineColLoc>(moduleOp.getLoc());
  if (loc) {
    filename = loc.getFilename();
    startLine = loc.getLine();
    startCol = loc.getColumn();
  }

  std::vector<Token> tokens;
  if (mlir::failed(lex(moduleOp->getContext(), program, filename, startLine,
                       startCol, tokens))) {

    return mlir::failure();
  }

  Parser parser(tokens, moduleOp);
  if (mlir::failed(parser.parse())) {
    return mlir::failure();
  }

  return mlir::success();
}

} // namespace graphalg
