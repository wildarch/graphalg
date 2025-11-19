# GraphAlg: A Embeddable Language for Writing Graph Algorithm in Linear Algebra
This repository contains code related to the GraphAlg language:
- `codemirror-lang-graphalg`: Language Support for Codemirror
- `compiler/`: The GraphAlg compiler
- `playground/`: The GraphAlg online playground
- `spec/`: The GraphAlg Language Specification

## Building
This assumes you are using the provided [devcontainer](https://containers.dev/) development environment.

```bash
npm --workspace=codemirror-lang-graphalg install
npm --workspace=codemirror-lang-graphalg run prepare

./compiler/configure.sh
cmake --build ./compiler/build
cmake --build ./compiler/build --target check

./compiler/configure-wasm.sh
cmake --build ./compiler/build-wasm

npm --workspace=playground install
# Or npm --workspace=playground run watch to rebuild automatically upon edit.
npm --workspace=playground run prepare

bundle install
# Or bundle exec jekyll serve to start a server
bundle exec jekyll build
```
