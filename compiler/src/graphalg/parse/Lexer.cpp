#include <optional>

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/StringSwitch.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>

#include "graphalg/parse/Lexer.h"

namespace graphalg {

llvm::StringLiteral Token::kindName(Kind k) {
  switch (k) {
#define GA_CASE(X)                                                             \
  case X:                                                                      \
    return #X;

    GRAPHALG_ENUM_TOKEN_KIND(GA_CASE)
#undef GA_CASE
  }
}

static std::optional<Token::Kind> tokenForKeyword(llvm::StringRef s) {
  auto token = llvm::StringSwitch<Token::Kind>(s)
                   .Case("func", Token::FUNC)
                   .Case("return", Token::RETURN)
                   .Case("for", Token::FOR)
                   .Case("in", Token::IN)
                   .Case("until", Token::UNTIL)
                   .Case("true", Token::TRUE)
                   .Case("false", Token::FALSE)
                   .Default(Token::END_OF_FILE);
  if (token == Token::END_OF_FILE) {
    return std::nullopt;
  } else {
    return token;
  }
}

Token Lexer::nextToken() {
  eatWhitespace();
  if (!cur()) {
    return Token{Token::END_OF_FILE, currentLocation()};
  }

  if (llvm::isAlpha(*cur())) {
    // Identifier [a-zA-Z][a-zA-Z0-9_']*
    auto loc = currentLocation();
    auto start = _offset;
    eat();
    while (cur() && (llvm::isAlnum(*cur()) || cur() == '_' || cur() == '\'')) {
      eat();
    }

    auto end = _offset;
    auto body = _buffer.slice(start, end);
    if (auto keyword = tokenForKeyword(body)) {
      return Token{*keyword, loc, body};
    }

    return Token{Token::IDENT, loc, body};
  }

  if (llvm::isDigit(*cur())) {
    // Number
    auto loc = currentLocation();
    auto start = _offset;
    eat();
    while (cur() && llvm::isDigit(*cur())) {
      eat();
    }

    auto kind = Token::INT;
    if (cur() == '.') {
      kind = Token::FLOAT;
      eat();
      while (cur() && llvm::isDigit(*cur())) {
        eat();
      }
    }

    auto end = _offset;
    auto body = _buffer.slice(start, end);
    return Token{kind, loc, body};
  }

  auto two = peek(2);
  auto loc = currentLocation();
#define TWO(c, t)                                                              \
  if (two == "->") {                                                           \
    eat();                                                                     \
    eat();                                                                     \
    return Token{Token::t, loc, *two};                                         \
  }

  TWO("->", ARROW)
  TWO("+=", ACCUM)
  TWO("==", EQUAL)
  TWO("!=", NOT_EQUAL)
  TWO("<=", LEQ)
  TWO(">=", GEQ)
#undef TWO

  auto one = _buffer.substr(_offset, 1);
#define ONE(c, t)                                                              \
  if (tryEat(c)) {                                                             \
    return Token{Token::t, loc, one};                                          \
  }

  ONE('(', LPAREN)
  ONE(')', RPAREN)
  ONE('{', LBRACE)
  ONE('}', RBRACE)
  ONE('[', LSBRACKET)
  ONE(']', RSBRACKET)
  ONE('<', LANGLE)
  ONE('>', RANGLE)
  ONE(':', COLON)
  ONE(',', COMMA)
  ONE('.', DOT)
  ONE(';', SEMI)
  ONE('+', PLUS)
  ONE('-', MINUS)
  ONE('*', TIMES)
  ONE('/', DIVIDE)
  ONE('=', ASSIGN)
  ONE('!', NOT)
#undef ONE

  mlir::emitError(currentLocation())
      << "invalid input character '" << *cur() << "'";
  return Token{Token::INVALID, currentLocation(), _buffer.substr(_offset, 1)};
}

Lexer::Lexer(mlir::MLIRContext *ctx, llvm::StringRef buffer,
             llvm::StringRef filename)
    : _filename(mlir::StringAttr::get(ctx, filename)), _buffer(buffer) {}

mlir::LogicalResult Lexer::lex(std::vector<Token> &tokens) {
  while (true) {
    auto token = nextToken();
    if (token.type == Token::END_OF_FILE) {
      break;
    } else if (token.type == Token::INVALID) {
      return mlir::failure();
    }

    tokens.push_back(token);
  }

  return mlir::success();
}

} // namespace graphalg
