#include <utility>
#include <vector>

#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/ScopedHashTable.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/StringSaver.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/Builders.h>
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
  // TODO: Can we always have this?
  const DimMapper *_dimMapper = nullptr;

  std::string _type;

  void formatScalar(mlir::Type t);
  void formatColumnVector(MatrixType t);
  void formatMatrix(MatrixType t);

public:
  TypeFormatter(const DimMapper *dimMapper) : _dimMapper(dimMapper) {}

  void format(mlir::Type t);

  std::string take() { return std::move(_type); }
};

class Parser {
private:
  llvm::ArrayRef<Token> _tokens;
  std::size_t _offset = 0;

  mlir::ModuleOp _module;
  mlir::OpBuilder _builder;

  /*
  llvm::BumpPtrAllocator _stringAllocator;
  llvm::UniqueStringSaver _stringPool;
  */

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

  mlir::ParseResult eatOrError(Token::Kind kind) {
    if (cur().type == kind) {
      _offset++;
      return mlir::success();
    } else {
      return mlir::emitError(cur().loc) << "expected " << Token::kindName(kind);
    }
  }

  std::string typeToString(mlir::Type t);

  mlir::LogicalResult assign(mlir::Location loc, llvm::StringRef name,
                             mlir::Value value);

  mlir::ParseResult parseIdent(llvm::StringRef &s);
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

  if (_dimMapper) {
    // Rows
    _type += _dimMapper->getName(t.getRows());
    _type += ", ";
  }

  formatScalar(t.getSemiring());
  _type += ">";
}

void TypeFormatter::formatMatrix(MatrixType t) {
  if (t.isColumnVector()) {
    return formatColumnVector(t);
  }

  _type += "Matrix<";

  if (_dimMapper) {
    // Rows
    _type += _dimMapper->getName(t.getRows());
    _type += ", ";

    // Columns
    _type += _dimMapper->getName(t.getCols());
    _type += ", ";
  }

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
  TypeFormatter fmt(&_dimMapper);
  fmt.format(type);
  return fmt.take();
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

mlir::ParseResult Parser::parseIdent(llvm::StringRef &s) {
  if (cur().type != Token::IDENT) {
    return mlir::emitError(cur().loc) << "expected identifier";
  }

  s = cur().body;
  return eatOrError(Token::IDENT);
}

mlir::ParseResult Parser::parseType(mlir::Type &t) {
  auto *ctx = _builder.getContext();
  if (auto ring = tryParseSemiring()) {
    t = ring;
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
    return mlir::success();
  } else if (cur().type == Token::IDENT) {
    t = _dimMapper.getOrAllocate(cur().body);
    return mlir::success();
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

  // TODO: parse body

  if (eatOrError(Token::RBRACE)) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult Parser::parse() {
  auto *ctx = _builder.getContext();
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
