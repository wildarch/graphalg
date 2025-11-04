import { parser } from "./syntax.grammar"
import { LRLanguage, LanguageSupport, indentNodeProp, foldNodeProp, foldInside, delimitedIndent } from "@codemirror/language"
import { styleTags, tags as t } from "@lezer/highlight"

export const GraphAlgLanguage = LRLanguage.define({
  name: "graphalg",
  parser: parser.configure({
    props: [
      indentNodeProp.add({
        Block: delimitedIndent({ closing: "}" }),
      }),
      foldNodeProp.add({
        Block: foldInside,
      }),
      styleTags({
        // https://lezer.codemirror.net/docs/ref/#highlight.tags
        Func: t.definitionKeyword,
        // Types
        Semiring: t.typeName,
        Matrix: t.typeName,
        Vector: t.typeName,
        // Dim
        DimOne: t.keyword,
        For: t.controlKeyword,
        Return: t.keyword,
        Until: t.controlKeyword,

        // Expressions
        Property: t.propertyName,
        Comparator: t.compareOperator,
        ZeroExpr: t.keyword,
        OneExpr: t.keyword,
        Cast: t.keyword,
        BuiltInFunc: t.keyword,

        // Tokens
        Ident: t.variableName,
        Boolean: t.bool,
        Number: t.number,
        Comment: t.lineComment,
        Dot: t.operator,
        Plus: t.arithmeticOperator,
        Minus: t.arithmeticOperator,
        Times: t.arithmeticOperator,
        Div: t.arithmeticOperator,
        Not: t.logicOperator,
        LBrace: t.brace,
        RBrace: t.brace,
      })
    ]
  }),
  languageData: {
    commentTokens: { line: "//" }
  }
})

export function GraphAlg() {
  return new LanguageSupport(GraphAlgLanguage)
}
