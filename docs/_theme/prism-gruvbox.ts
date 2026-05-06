import type { PrismTheme } from "prism-react-renderer";

const gruvboxLight: PrismTheme = {
  plain: {
    color: "#443a32",
    backgroundColor: "#eee8dc",
  },
  styles: [
    { types: ["comment", "prolog", "doctype", "cdata"], style: { color: "#8a7e72", fontStyle: "italic" } },
    { types: ["punctuation"], style: { color: "#443a32" } },
    { types: ["property", "tag", "boolean", "constant", "symbol", "deleted"], style: { color: "#9d0006" } },
    { types: ["number"], style: { color: "#8f3f71" } },
    { types: ["selector", "attr-name", "string", "char", "builtin", "inserted"], style: { color: "#79740e" } },
    { types: ["operator", "entity", "url"], style: { color: "#477a5b" } },
    { types: ["atrule", "attr-value", "keyword"], style: { color: "#9d0006" } },
    { types: ["function"], style: { color: "#79740e" } },
    { types: ["regex", "important", "variable"], style: { color: "#b57614" } },
    { types: ["class-name", "maybe-class-name"], style: { color: "#b57614" } },
    { types: ["namespace"], style: { color: "#b57614" } },
    { types: ["italic"], style: { fontStyle: "italic" } },
    { types: ["bold"], style: { fontWeight: "bold" } },
  ],
};

const gruvboxDark: PrismTheme = {
  plain: {
    color: "#ddd0b8",
    backgroundColor: "#2c2a26",
  },
  styles: [
    { types: ["comment", "prolog", "doctype", "cdata"], style: { color: "#928677", fontStyle: "italic" } },
    { types: ["punctuation"], style: { color: "#ddd0b8" } },
    { types: ["property", "tag", "boolean", "constant", "symbol", "deleted"], style: { color: "#e36d6b" } },
    { types: ["number"], style: { color: "#d3869b" } },
    { types: ["selector", "attr-name", "string", "char", "builtin", "inserted"], style: { color: "#a9b665" } },
    { types: ["operator", "entity", "url"], style: { color: "#89b482" } },
    { types: ["atrule", "attr-value", "keyword"], style: { color: "#e36d6b" } },
    { types: ["function"], style: { color: "#a9b665" } },
    { types: ["regex", "important", "variable"], style: { color: "#deb24e" } },
    { types: ["class-name", "maybe-class-name"], style: { color: "#deb24e" } },
    { types: ["namespace"], style: { color: "#deb24e" } },
    { types: ["italic"], style: { fontStyle: "italic" } },
    { types: ["bold"], style: { fontWeight: "bold" } },
  ],
};

export { gruvboxLight, gruvboxDark };
