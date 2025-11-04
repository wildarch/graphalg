# GraphAlg: A Embeddable Language for Writing Graph Algorithm in Linear Algebra
This repository contains code related to the GraphAlg language:
- `codemirror-lang-graphalg`: Language Support for Codemirror
- `compiler/`: The GraphAlg compiler
- `website/`: The GraphAlg online playground
- `spec/`: The GraphAlg Language Specification

## Building
This assumes you are using the provided [devcontainer](https://containers.dev/) development environment.

```bash
npm --workspace=codemirror-lang-graphalg install
npm --workspace=codemirror-lang-graphalg run prepare

./compiler/configure.sh
cmake --build ./compiler/build
cmake --build ./compiler/build --target check

npm --workspace=website install
npm --workspace=website run prepare
# To start a simple server to test the website:
# (cd website/ && python3 -m http.server)

bundle install
# Or bundle exec jekyll serve to start a server
bundle exec jekyll build
```
