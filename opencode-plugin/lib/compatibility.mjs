import { execFile } from "node:child_process";
import { promisify } from "node:util";

const execFileAsync = promisify(execFile);
export const MINIMUM_OPENCODE_VERSION = "1.1.13";
export const DISPOSE_OPENCODE_VERSION = "1.15.11";

export function parseVersion(text) {
  const match = String(text || "").match(/(?:^|\s|v)(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z.-]+))?(?:\+[0-9A-Za-z.-]+)?(?=$|\s)/);
  if (!match) return null;
  const version = match.slice(1, 4).map(Number);
  Object.defineProperty(version, "prerelease", { value: Boolean(match[4]) });
  return version;
}

export function versionAtLeast(actual, minimum = MINIMUM_OPENCODE_VERSION) {
  const a = Array.isArray(actual) ? actual : parseVersion(actual);
  const b = parseVersion(minimum);
  if (!a || !b) return null;
  for (let i = 0; i < 3; i += 1) {
    if (a[i] !== b[i]) return a[i] > b[i];
  }
  return !a.prerelease || Boolean(b.prerelease);
}

export async function detectCompatibility(run = execFileAsync) {
  try {
    const { stdout, stderr } = await run("opencode", ["--version"], { windowsHide: true, timeout: 3000 });
    const version = parseVersion(`${stdout} ${stderr}`);
    return { supported: versionAtLeast(version), hasDispose: versionAtLeast(version, DISPOSE_OPENCODE_VERSION), version, minimum: MINIMUM_OPENCODE_VERSION };
  } catch (error) {
    return { supported: null, hasDispose: null, version: null, minimum: MINIMUM_OPENCODE_VERSION, error: error.message };
  }
}
