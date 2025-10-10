import {EditorView, basicSetup} from "codemirror"
import { vim } from "@replit/codemirror-vim"
import {keymap} from "@codemirror/view"
import {indentWithTab} from "@codemirror/commands"
import {GraphAlg} from "codemirror-lang-graphalg"

/*
import {tags} from "@lezer/highlight"
import {HighlightStyle, syntaxHighlighting} from "@codemirror/language"
const myHighlightStyle = HighlightStyle.define([
  {tag: tags.logicOperator, color: "#fc6"}
])
*/

let editor = new EditorView({
  extensions: [
    vim(), 
    keymap.of([indentWithTab]),
    basicSetup,
    GraphAlg()
  ],
  parent: document.body
})