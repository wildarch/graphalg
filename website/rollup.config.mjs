import {nodeResolve} from "@rollup/plugin-node-resolve"
import {lezer} from "@lezer/generator/rollup"

export default {
  input: "./editor.mjs",
  output: {
    file: "./editor.bundle.js",
    format: "iife"
  },
  plugins: [nodeResolve(), lezer()]
}