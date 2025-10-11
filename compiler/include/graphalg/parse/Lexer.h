#pragma once

#include <cassert>
#include <optional>
#include <vector>

#include <llvm/ADT/StringExtras.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/Diagnostics.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Support/LLVM.h>

namespace graphalg {

#define GRAPHALG_ENUM_TOKEN_KIND(XX)                                           \
  XX(INVALID)     /* SPECIAL: For inputs that could not be tokenized. */       \
  XX(END_OF_FILE) /* SPECIAL: End of file marker. */                           \
  XX(IDENT)                                                                    \
  XX(INT)                                                                      \
  XX(FLOAT)                                                                    \
  /* Keywords */                                                               \
  XX(FUNC)                                                                     \
  XX(RETURN)                                                                   \
  XX(FOR)                                                                      \
  XX(IN)                                                                       \
  XX(UNTIL)                                                                    \
  /* Punctuation */                                                            \
  XX(LPAREN)    /* ( */                                                        \
  XX(RPAREN)    /* ) */                                                        \
  XX(LBRACE)    /* { */                                                        \
  XX(RBRACE)    /* } */                                                        \
  XX(LSBRACKET) /* [ */                                                        \
  XX(RSBRACKET) /* ] */                                                        \
  XX(LANGLE)    /* < */                                                        \
  XX(RANGLE)    /* > */                                                        \
  XX(COLON)     /* : */                                                        \
  XX(COMMA)     /* , */                                                        \
  XX(DOT)       /* . */                                                        \
  XX(SEMI)      /* ; */                                                        \
  /* Operators */                                                              \
  XX(PLUS)      /* + */                                                        \
  XX(MINUS)     /* - */                                                        \
  XX(TIMES)     /* * */                                                        \
  XX(DIVIDE)    /* / */                                                        \
  XX(ASSIGN)    /* = */                                                        \
  XX(NOT)       /* ! */                                                        \
  XX(ARROW)     /* -> */                                                       \
  XX(ACCUM)     /* += */                                                       \
  XX(EQUAL)     /* == */                                                       \
  XX(NOT_EQUAL) /* != */                                                       \
  XX(LEQ)       /* <= */                                                       \
  XX(GEQ)       /* >= */                                                       \
  /* Literals */                                                               \
  XX(TRUE)                                                                     \
  XX(FALSE)

struct Token {
  enum Kind {
#define GA_CASE(X) X,
    GRAPHALG_ENUM_TOKEN_KIND(GA_CASE)
#undef GA_CASE
  };

  static llvm::StringLiteral kindName(Kind k);

  Kind type;
  mlir::Location loc;
  llvm::StringRef body = "";
};

class Lexer {
private:
  mlir::StringAttr _filename;
  llvm::StringRef _buffer;
  std::size_t _offset = 0;
  std::size_t _line = 1;
  std::size_t _col = 1;

  std::optional<char> cur() {
    if (_offset < _buffer.size()) {
      return _buffer[_offset];
    } else {
      return std::nullopt;
    }
  }

  std::optional<llvm::StringRef> peek(std::size_t n) {
    if (_offset + n < _buffer.size()) {
      return _buffer.substr(_offset, n);
    } else {
      return std::nullopt;
    }
  }

  mlir::Location currentLocation() {
    return mlir::FileLineColLoc::get(_filename, _line, _col);
  }

  void eat() {
    if (cur() == '\n') {
      _col = 1;
      _line += 1;
    } else {
      _col += 1;
    }

    _offset++;
  }

  bool tryEat(char c) {
    if (cur() == c) {
      eat();
      return true;
    } else {
      return false;
    }
  }

  void eatWhitespace() {
    while (true) {
      if (cur() && llvm::isSpace(*cur())) {
        // Simple whitespace
        eat();
        continue;
      } else if (peek(2) == "//") {
        // Line comment
        while (cur() != '\n') {
          eat();
        }

        assert(cur() == '\n');
        eat();
        continue;
      }

      break;
    }
  }

  Token nextToken();

public:
  Lexer(mlir::MLIRContext *ctx, llvm::StringRef buffer,
        llvm::StringRef filename = "<unknown>");

  mlir::LogicalResult lex(std::vector<Token> &tokens);
};

} // namespace graphalg
