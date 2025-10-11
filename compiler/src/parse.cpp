#include <iostream>
#include <string>
#include <vector>

#include <graphalg/parse/Lexer.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/MLIRContext.h>

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <path>\n";
    return 1;
  }

  mlir::MLIRContext ctx;

  char *filename = argv[1];
  auto file = llvm::sys::fs::openNativeFileForRead(filename);
  if (auto err = file.takeError()) {
    llvm::errs() << "Cannot open file: '" << filename << "': " << err << "\n";
    return 1;
  }

  llvm::SmallVector<char> buffer;
  if (auto err = llvm::sys::fs::readNativeFileToEOF(*file, buffer)) {
    llvm::errs() << "Cannot read file: '" << filename << "': " << err << "\n";
    return 1;
  }

  std::string bufferString(buffer.data(), buffer.size());
  graphalg::Lexer lexer(&ctx, bufferString, filename);
  std::vector<graphalg::Token> tokens;
  if (mlir::failed(lexer.lex(tokens))) {
    llvm::errs() << "Error parsing input\n";
    return 1;
  }

  for (auto token : tokens) {
    llvm::outs() << "token " << graphalg::Token::kindName(token.type) << " '"
                 << token.body << "'\n";
  }

  return 0;
}
