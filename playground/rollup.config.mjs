import { nodeResolve } from "@rollup/plugin-node-resolve"
import { lezer } from "@lezer/generator/rollup"
import url from '@rollup/plugin-url';

const urlOptions = {
  include: '**/*.wasm',
};

export default {
  input: "./editor.mjs",
  output: {
    file: "./editor.bundle.js",
    format: "iife"
  },
  plugins: [nodeResolve(), lezer(), url(urlOptions)]
}
