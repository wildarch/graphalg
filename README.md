# GraphAlg: An Embeddable Language for Writing Graph Algorithms in Linear Algebra

GraphAlg is a domain-specific language for graph algorithms based on linear algebra,
designed to be embedded in databases. Try it in the browser via the
[Playground](https://wildarch.dev/graphalg/playground) or learn the language with the
[interactive tutorial](https://wildarch.dev/graphalg/tutorial).

This repository contains code related to the GraphAlg language:
- `codemirror-lang-graphalg`: Language Support for Codemirror
- `compiler/`: The GraphAlg compiler.
  Includes the parser, lowering to GraphAlg Core, high-level optimizations, and a reference backend for executing algorithms.
- `playground/`: The GraphAlg online playground
- `spec/`: The GraphAlg Language Specification
- `tutorial/`: A tutorial for new GraphAlg users

## Citation

If you use GraphAlg in your work, please cite the software via the metadata in
[`CITATION.cff`](./CITATION.cff) (or the Zenodo DOI for a specific release).

## Acknowledgement

This work was supported by the European Union's Horizon Europe research and innovation
programme under grant agreement **No. 101058573** ([SciLake](https://scilake.eu/)).

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
