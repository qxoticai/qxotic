import { gruvboxLight, gruvboxDark } from "./docs/_theme/prism-gruvbox";
import type { Config } from "@docusaurus/types";
import type * as Preset from "@docusaurus/preset-classic";
import remarkSnippets from "./docs/_theme/remark/snippets";

const config: Config = {
  title: "Qxotic",
  tagline: "Java libraries for LLM inference and model formats",
  favicon: "docs/img/favicon.svg",

  future: {
    v4: true,
  },

  url: "https://qxotic.ai",
  baseUrl: "/",

  organizationName: "qxoticai",
  projectName: "qxotic",

  onBrokenLinks: "warn",
  onBrokenAnchors: "warn",

  i18n: {
    defaultLocale: "en",
    locales: ["en"],
  },

  themes: [
    ["@easyops-cn/docusaurus-search-local", {
      docsRouteBasePath: "/",
      indexDocs: true,
    }],
  ],

  presets: [
    [
      "classic",
      {
        docs: {
          path: "docs",
          routeBasePath: "/",
          sidebarPath: false,
          editUrl: ({ docPath }) =>
            `https://github.com/qxoticai/qxotic/edit/main/docs/${docPath}`,
          beforeDefaultRemarkPlugins: [remarkSnippets],
        },
        blog: false,
        theme: {
          customCss: "./docs/_theme/custom.css",
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    colorMode: {
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: "Qxotic",
      logo: {
        alt: "Qxotic",
        src: "docs/img/logo.svg",
      },
      items: [
        {
          to: "/safetensors",
          position: "left",
          label: "Safetensors",
        },
        {
          to: "/json",
          position: "left",
          label: "JSON",
        },
        {
          to: "/gguf",
          position: "left",
          label: "GGUF",
        },
        {
          to: "/toknroll",
          position: "left",
          label: "Tok'n'Roll",
        },
        {
          type: "search",
          position: "right",
        },
        {
          href: "https://javadoc.io/doc/com.qxotic",
          label: "Javadoc",
          position: "right",
        },
        {
          href: "https://github.com/qxoticai/qxotic",
          label: "GitHub",
          position: "right",
        },
      ],
    },
    footer: {
      links: [
        {
          items: [
            {
              html: '<a href="https://github.com/qxoticai/qxotic" target="_blank" rel="noopener" aria-label="GitHub"><svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg></a>',
            },
            {
              html: '<a href="https://x.com/qxoticai" target="_blank" rel="noopener" aria-label="X"><svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M18.244 2.25h3.308l-7.227 8.26 8.502 11.24H16.17l-5.214-6.817L4.99 21.75H1.68l7.73-8.835L1.254 2.25H8.08l4.713 6.231zm-1.161 17.52h1.833L7.084 4.126H5.117z"/></svg></a>',
            },
            {
              html: '<a href="https://bsky.app/profile/qxotic.ai" target="_blank" rel="noopener" aria-label="Bluesky"><svg viewBox="0 0 568 501" width="24" height="21" fill="currentColor"><path d="M123.121 33.664C188.241 82.553 258.281 181.651 284 234.873c25.719-53.222 95.759-152.32 144.879-201.21C491.866-1.611 568-28.906 568 57.947c0 17.345-9.945 145.713-15.778 166.555-20.275 72.453-94.155 90.933-159.875 79.748 114.875 19.551 144.097 84.311 80.986 149.071-119.86 122.992-172.324-30.859-185.754-70.277-2.462-7.227-3.614-10.608-3.631-7.733-.017-2.875-1.169-.456-3.631 7.733-13.43 39.418-65.894 193.269-185.754 70.277-63.111-64.76-33.889-119.52 80.986-149.071-65.72 11.185-139.6-7.295-159.875-79.748C9.945 203.66 0 75.291 0 57.947 0-28.906 76.134-1.611 123.121 33.664Z"/></svg></a>',
            },
          ],
        },
      ],
      copyright: `Accelerating (on) the JVM.\nBuilt in 🇨🇭 Switzerland · ${new Date().getFullYear()} Quixotic AI`,
    },
    prism: {
      theme: gruvboxLight,
      darkTheme: gruvboxDark,
      additionalLanguages: ["java", "bash"],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
