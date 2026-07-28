import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { cp, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { fileURLToPath, pathToFileURL } from "node:url";
import { classifyNpmView } from "../scripts/check-published.mjs";

const packageRoot = fileURLToPath(new URL("..", import.meta.url));
function runNpm(args, options) {
  if (process.platform !== "win32") return spawnSync("npm", args, options);
  const command = `npm.cmd ${args.map((value) => /[\s&|<>^]/.test(String(value)) ? `"${String(value).replaceAll('"', '""')}"` : String(value)).join(" ")}`;
  return spawnSync(process.env.ComSpec, ["/d", "/s", "/c", command], options);
}

function workflowStep(workflow, name) {
  const start = workflow.indexOf(`      - name: ${name}`);
  assert.notEqual(start, -1, `missing workflow step: ${name}`);
  const next = workflow.indexOf("\n      - name:", start + 1);
  return workflow.slice(start, next === -1 ? workflow.length : next);
}

function runBlock(step) {
  const marker = "        run: |\n";
  const start = step.indexOf(marker);
  assert.notEqual(start, -1, "workflow step is missing a shell run block");
  return step.slice(start + marker.length).split("\n").map((line) => line.startsWith("          ") ? line.slice(10) : line).join("\n").trim();
}

async function removeTemporary(directory) {
  for (let attempt = 0; attempt < 5; attempt += 1) {
    try { await rm(directory, { recursive: true, force: true }); return; }
    catch (error) {
      if (!['EBUSY', 'EPERM'].includes(error.code) || attempt === 4) throw error;
      await new Promise((resolve) => setTimeout(resolve, 20));
    }
  }
}

test("npm publication probe distinguishes exists, E404, and registry failure", () => {
  assert.equal(classifyNpmView(0, "0.1.0"), true);
  assert.equal(classifyNpmView(1, "npm error code E404"), false);
  assert.throws(() => classifyNpmView(1, "npm error code E500"), /publication was not attempted/);
  assert.throws(() => classifyNpmView(null, "network unavailable"), /registry lookup failed/);
});

test("release contract uses token publishing for the GitCode source repository", async () => {
  const manifest = JSON.parse(await readFile(join(packageRoot, "package.json"), "utf8"));
  const workflow = await readFile(join(packageRoot, "..", ".github", "workflows", "opencode-plugin-release.yml"), "utf8");
  assert.equal(manifest.repository.url, "https://gitcode.com/opengauss/oGMemory.git");
  assert.equal(manifest.publishConfig.provenance, undefined);
  assert.doesNotMatch(workflow, /--provenance|id-token:\s*write/);
  assert.match(workflow, /NPM_TOKEN:\s*\$\{\{\s*secrets\.NPM_TOKEN\s*\}\}/);
  assert.match(workflow, /pull_request:/);
  const publish = workflowStep(workflow, "Publish package");
  assert.match(publish, /if:\s*steps\.published\.outputs\.exists == 'false'/);
  const block = runBlock(publish);
  assert.ok(block.indexOf("exit 1") < block.indexOf("npm config set"));
  assert.ok(block.indexOf("npm config set") < block.indexOf("npm publish --access public"));
});

test("release publish shell fails without a token and authenticates before publish", async (t) => {
  const probe = spawnSync("bash", ["--version"], { encoding: "utf8" });
  if (probe.error?.code === "ENOENT") { t.skip("bash is unavailable on this platform"); return; }
  assert.equal(probe.status, 0, probe.stderr);
  const workflow = await readFile(join(packageRoot, "..", ".github", "workflows", "opencode-plugin-release.yml"), "utf8");
  const block = runBlock(workflowStep(workflow, "Publish package"));
  const directory = await mkdtemp(join(tmpdir(), "ogmemory-release-shell-"));
  try {
    const harness = `npm() { printf '%s\\n' "$*" >> npm-calls.log; }\n${block}`;
    const execute = (script) => spawnSync("bash", ["-c", `echo ${Buffer.from(script).toString("base64")} | base64 -d | bash`], { cwd: directory, encoding: "utf8" });
    const missing = execute(`unset NPM_TOKEN\n${harness}`);
    assert.notEqual(missing.status, 0);
    await assert.rejects(readFile(join(directory, "npm-calls.log"), "utf8"), (error) => error.code === "ENOENT");

    const present = execute(`export NPM_TOKEN=test-token\n${harness}`);
    assert.equal(present.status, 0, present.stderr);
    assert.deepEqual((await readFile(join(directory, "npm-calls.log"), "utf8")).trim().split(/\r?\n/), [
      "config set //registry.npmjs.org/:_authToken test-token",
      "publish --access public",
    ]);
  } finally { await removeTemporary(directory); }
});

test("packed npm artifact installs and exposes the plugin factory", async () => {
  const directory = await mkdtemp(join(tmpdir(), "ogmemory-pack-"));
  try {
    const packed = runNpm(["pack", "--json", "--pack-destination", directory], { cwd: packageRoot, encoding: "utf8", timeout: 30000 });
    assert.equal(packed.status, 0, packed.stderr || packed.stdout);
    const artifact = JSON.parse(packed.stdout)[0];
    assert.equal(artifact.files.some(({ path }) => path.startsWith("wrappers/")), false);
    assert.equal(artifact.files.some(({ path }) => path.startsWith("tests/")), false);
    assert.equal(artifact.files.some(({ path }) => path.startsWith("scripts/")), false);
    const filename = artifact.filename;
    const project = join(directory, "consumer");
    await mkdir(project);
    await writeFile(join(project, "package.json"), JSON.stringify({ type: "module", dependencies: { "@opengauss/ogmemory-opencode-plugin": `file:${join(directory, filename)}` } }));
    const installed = runNpm(["install", "--ignore-scripts", "--no-package-lock"], { cwd: project, encoding: "utf8", timeout: 30000 });
    assert.equal(installed.status, 0, installed.stderr || installed.stdout);
    await writeFile(join(project, "probe.mjs"), 'import plugin, { OgMemoryPlugin } from "@opengauss/ogmemory-opencode-plugin"; if (plugin !== OgMemoryPlugin || typeof plugin !== "function") process.exit(2);');
    const probe = spawnSync(process.execPath, ["probe.mjs"], { cwd: project, encoding: "utf8", timeout: 5000 });
    assert.equal(probe.status, 0, probe.stderr || probe.stdout);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test("documented source wrapper layout imports the copied plugin", async () => {
  const directory = await mkdtemp(join(tmpdir(), "ogmemory-source-"));
  try {
    const plugins = join(directory, "plugins");
    const target = join(plugins, "ogmemory");
    await mkdir(target, { recursive: true });
    await cp(join(packageRoot, "lib"), join(target, "lib"), { recursive: true });
    await writeFile(join(target, "index.mjs"), await readFile(join(packageRoot, "index.mjs")));
    await writeFile(join(target, "package.json"), await readFile(join(packageRoot, "package.json")));
    await writeFile(join(plugins, "ogmemory.js"), await readFile(join(packageRoot, "wrappers", "ogmemory.js")));
    const imported = await import(`${pathToFileURL(join(plugins, "ogmemory.js")).href}?test=${Date.now()}`);
    assert.equal(typeof imported.default, "function");
    assert.equal(imported.default, imported.OgMemoryPlugin);
  } finally { await rm(directory, { recursive: true, force: true }); }
});
