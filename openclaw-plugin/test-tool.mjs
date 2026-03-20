#!/usr/bin/env node
/**
 * Test script to verify og_memory_write tool works end-to-end
 */
import { register } from './index.ts';

const mockApi = {
  registerTool: ({ name, label, description, parameters, execute }) => {
    console.log(`✅ Tool registered: ${name}`);
    console.log(`   Label: ${label}`);
    console.log(`   Description: ${description}`);

    // Simulate a tool call
    const testParams = {
      content: "I like dark roast coffee in the morning",
      category: "preference",
      tags: ["food", "beverage", "morning"]
    };

    console.log(`\n🧪 Testing tool with params:`, JSON.stringify(testParams, null, 2));

    execute("test-call-" + Date.now(), testParams)
      .then(result => {
        console.log(`\n✅ Tool executed successfully:`);
        console.log(JSON.stringify(result, null, 2));
      })
      .catch(err => {
        console.error(`\n❌ Tool execution failed:`);
        console.error(err.message);
        process.exit(1);
      });
  },
  logger: {
    info: (msg) => console.log(`[INFO] ${msg}`),
    warn: (msg) => console.warn(`[WARN] ${msg}`),
    error: (msg) => console.error(`[ERROR] ${msg}`)
  },
  config: {
    plugins: {
      entries: {
        "og-memory": {
          config: {
            pythonPath: "/home/aaa/miniconda3/envs/py11/bin/python",
            projectRoot: "/data/Workspace2/oG-Memory"
          }
        }
      }
    }
  }
};

console.log("Loading og-memory plugin...\n");
register(mockApi);
