import { appendFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { pathToFileURL } from "node:url";

export function classifyNpmView(status, output = "") {
  if (status === 0) return true;
  if (/\bE404\b/.test(output)) return false;
  throw new Error(`npm registry lookup failed with exit status ${status}; publication was not attempted`);
}

function main() {
  const name = process.env.PACKAGE_NAME;
  const version = process.env.PACKAGE_VERSION;
  const outputFile = process.env.GITHUB_OUTPUT;
  if (!name || !version || !outputFile) throw new Error("PACKAGE_NAME, PACKAGE_VERSION, and GITHUB_OUTPUT are required");
  const result = spawnSync("npm", ["view", `${name}@${version}`, "version", "--registry=https://registry.npmjs.org"], { encoding: "utf8", shell: process.platform === "win32" });
  const exists = classifyNpmView(result.status, `${result.stdout || ""}\n${result.stderr || ""}`);
  appendFileSync(outputFile, `exists=${exists}\n`);
  if (exists) console.log(`::notice::${name}@${version} already exists; skipping publication.`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  try { main(); }
  catch (error) { console.error(error.message); process.exitCode = 1; }
}
