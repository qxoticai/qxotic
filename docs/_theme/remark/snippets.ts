/**
 * Remark plugin that extracts tagged snippets from source files.
 *
 * Usage in markdown:
 *   ```snippet path="safetensors/src/test/.../Snippets.java" tag="read-path"
 *   ```
 *
 * The plugin reads the file (path relative to the site root / project root),
 * finds the region between:
 *   // --8<-- [start:TAG]
 *   // --8<-- [end:TAG]
 * and replaces the code block with a standard java code block.
 *
 * If path starts with "safetensors/", "json/", or "gguf/",
 * it resolves from the project root (where docusaurus.config.ts lives).
 * Otherwise falls back to vfile-relative resolution.
 */

import fs from "node:fs";
import path from "node:path";
import { visit } from "unist-util-visit";
import type { Plugin } from "unified";
import type { Code, Root } from "mdast";

const PROJECT_MODULES = ["safetensors", "json", "gguf", "toknroll"];

function resolvePath(filePath: string, vfileDir: string, siteDir: string): string {
  if (path.isAbsolute(filePath)) return filePath;
  const firstSegment = filePath.split(path.sep)[0];
  if (PROJECT_MODULES.includes(firstSegment)) {
    return path.resolve(siteDir, filePath);
  }
  return path.resolve(vfileDir, filePath);
}

function dedent(text: string): string {
  const lines = text.split("\n");
  const nonBlank = lines.filter((l) => l.trim().length > 0);
  if (nonBlank.length === 0) return text;
  const minIndent = Math.min(...nonBlank.map((l) => l.match(/^ */)?.[0].length ?? 0));
  if (minIndent === 0) return text;
  return lines.map((l) => (l.length === 0 ? l : l.slice(minIndent))).join("\n");
}

const remarkSnippets: Plugin<[], Root> = () => {
  return (tree, vfile) => {
    const vfileDir = (vfile as { dirname?: string }).dirname ?? ".";
    // vfileDir is e.g. /home/.../qxotic/docs/safetensors
    // siteDir is       /home/.../qxotic
    const match = vfileDir.match(/(.*?)\/docs/);
    const siteDir = match ? match[1] : vfileDir;

    visit(tree, "code", (node: Code) => {
      if (node.lang !== "snippet") return;

      const meta = node.meta ?? "";
      const pathMatch = meta.match(/path=["']([^"']+)["']/);
      const tagMatch = meta.match(/tag=["']([^"']+)["']/);

      if (!pathMatch || !tagMatch) return;

      const filePath = resolvePath(pathMatch[1], vfileDir, siteDir);
      const tag = tagMatch[1];

      let source: string;
      try {
        source = fs.readFileSync(filePath, "utf8");
      } catch {
        node.value = `// Error: file not found: ${filePath}`;
        node.lang = "java";
        return;
      }

      const startMarker = `[start:${tag}]`;
      const endMarker = `[end:${tag}]`;

      const lines = source.split("\n");
      let startIdx = -1;
      let endIdx = -1;

      for (let i = 0; i < lines.length; i++) {
        if (startIdx === -1 && lines[i].includes(startMarker)) {
          startIdx = i + 1;
        }
        if (startIdx !== -1 && lines[i].includes(endMarker)) {
          endIdx = i;
          break;
        }
      }

      if (startIdx === -1 || endIdx === -1) {
        node.value = `// Error: tag "${tag}" not found in ${filePath}`;
        node.lang = "java";
        return;
      }

      const snippet = lines.slice(startIdx, endIdx).join("\n");
      node.value = dedent(snippet).trimEnd();
      node.lang = "java";
    });
  };
};

export default remarkSnippets;
