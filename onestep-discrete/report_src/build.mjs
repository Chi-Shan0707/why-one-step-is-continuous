// Build a self-contained report.html: KaTeX pre-rendered (strict), KaTeX CSS + woff2 fonts and figures inlined.
//   node build.mjs <report.src.html> <out.html> <repo root>     (needs `npm install katex@0.16.11`)
import fs from "fs";
import path from "path";
import { createRequire } from "module";

const require = createRequire(path.join(process.cwd(), "x.js"));
const katex = require("katex");
const [src, out, root] = process.argv.slice(2);
const kdir = path.dirname(require.resolve("katex/dist/katex.min.css"));
const b64 = (f) => fs.readFileSync(f).toString("base64");

const tex = (s, displayMode) =>
  katex.renderToString(s.trim(), { displayMode, throwOnError: true, strict: "error" });

let html = fs.readFileSync(src, "utf8")
  .replace(/\\\[([\s\S]+?)\\\]/g, (_, s) => tex(s, true))
  .replace(/\\\(([\s\S]+?)\\\)/g, (_, s) => tex(s, false))
  .replace(/src="([^"]+\.png)"/g, (_, f) => `src="data:image/png;base64,${b64(path.join(root, f))}"`);

const css = fs.readFileSync(path.join(kdir, "katex.min.css"), "utf8").replace(
  /src:url\(fonts\/([^)]+\.woff2)\) format\("woff2"\)[^;}]*/g,
  (_, f) => `src:url(data:font/woff2;base64,${b64(path.join(kdir, "fonts", f))}) format("woff2")`);
html = html.replace("<!--KATEX_CSS-->", `<style>${css}</style>`);

if (/\\\(|\\\[/.test(html)) throw new Error("unrendered math delimiter left");
fs.writeFileSync(out, html);
console.log(`wrote ${out} (${(html.length / 1e6).toFixed(2)} MB)`);
