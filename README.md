# GraphAlg: A Embeddable Language for Writing Graph Algorithm in Linear Algebra
This repository contains code related to the GraphAlg language:
- `codemirror-lang-graphalg`: Language Support for Codemirror
- `compiler/`: The GraphAlg compiler.
  Includes the parser, lowering to GraphAlg Core, high-level optimizations, and a reference backend for executing algorithms.
- `playground/`: The GraphAlg online playground
- `spec/`: The GraphAlg Language Specification
- `tutorial/`: A tutorial for new GraphAlg users

## Building
This assumes you are using the provided [devcontainer](https://containers.dev/) development environment.

```bash
npm --workspace=codemirror-lang-graphalg install
npm --workspace=codemirror-lang-graphalg run prepare

./compiler/configure.sh
cmake --build ./compiler/build
cmake --build ./compiler/build --target check

npm --workspace=playground install
playground/cpp/configure-wasm.sh
cmake --build playground/cpp/build-wasm --target graphalg-playground
# Or npm --workspace=playground run watch to rebuild automatically upon edit.
npm --workspace=playground run prepare

bundle install
# Or bundle exec jekyll serve to start a server
bundle exec jekyll build
```
