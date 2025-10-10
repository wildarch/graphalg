import {GraphAlg} from "codemirror-lang-graphalg"

let tree = GraphAlg().language.parser.parse("func Test(a: Matrix<s, 1, int>) -> Vector<s, int> { return reduceRows(a); }")

let cursor = tree.cursor()
do {
  console.log(`Node ${cursor.name} from ${cursor.from} to ${cursor.to}`)
} while (cursor.next())