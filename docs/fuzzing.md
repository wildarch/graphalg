# Fuzzing the Parser
Create a new corpus from the parser tests:

```bash
export CORPUS_DIR=<some path>
mkdir $CORPUS_DIR
cp compiler/test/parse/*.gr $CORPUS_DIR/
cp compiler/test/parse-err/*.gr $CORPUS_DIR/

./compiler/configure.sh
cmake --build ./compiler/build --target fuzz-parser
./compiler/build/tools/fuzz-parser $CORPUS_DIR -max_len=1000 -jobs=8 -only_ascii=1
```
