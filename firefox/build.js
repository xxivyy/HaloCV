import * as esbuild from "esbuild";

const common = {
  bundle: true,
  sourcemap: false,
  target: "es2022",
};

await Promise.all([
  esbuild.build({
    ...common,
    entryPoints: ["src/background.ts"],
    outfile: "dist/background.js",
  }),

  esbuild.build({
    ...common,
    entryPoints: ["src/content.ts"],
    outfile: "dist/content.js",
  }),

  esbuild.build({
    ...common,
    entryPoints: ["src/popup.ts"],
    outfile: "dist/popup.js",
  }),
]);
