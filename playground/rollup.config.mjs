import { nodeResolve } from "@rollup/plugin-node-resolve"
import typescript from '@rollup/plugin-typescript';
import { lezer } from "@lezer/generator/rollup"
import url from '@rollup/plugin-url';

const urlOptions = {
  include: '**/*.wasm',
};

export default {
  input: "./editor.ts",
  output: {
    file: "./editor.bundle.js",
    format: "iife"
  },
  plugins: [nodeResolve(), typescript(), lezer(), url(urlOptions)]
}
