import test from "node:test";
import assert from "node:assert/strict";
import { classifyNpmView } from "../scripts/check-published.mjs";

test("npm publication probe distinguishes exists, E404, and registry failure", () => {
  assert.equal(classifyNpmView(0, "0.1.0"), true);
  assert.equal(classifyNpmView(1, "npm error code E404"), false);
  assert.throws(() => classifyNpmView(1, "npm error code E500"), /publication was not attempted/);
  assert.throws(() => classifyNpmView(null, "network unavailable"), /registry lookup failed/);
});
