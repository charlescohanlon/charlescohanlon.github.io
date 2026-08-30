import { defineConfig } from "astro/config";
import mdx from "@astrojs/mdx";
import sitemap from "@astrojs/sitemap";
import tailwindcss from "@tailwindcss/vite";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import remarkObsidianImages from "./src/plugins/remark-obsidian-images.mjs";
import rehypeSidenotes from "./src/plugins/rehype-sidenotes.mjs";
import rehypeGlueMathPunctuation from "./src/plugins/rehype-glue-math-punctuation.mjs";

// https://astro.build/config
export default defineConfig({
  site: "https://charlescohanlon.com",
  // Flat files (blog.html, not blog/index.html): GitHub Pages serves them
  // extensionless, so URLs don't carry a trailing slash.
  trailingSlash: "never",
  build: { format: "file" },
  redirects: {
    "/portfolio": "/research",
    "/publications": "/research",
  },
  integrations: [mdx(), sitemap()],
  vite: {
    plugins: [tailwindcss()],
  },
  markdown: {
    remarkRehype: {
      // U+FE0E pins the arrow to text presentation — a bare U+21A9
      // renders as an emoji on iOS.
      footnoteBackContent: "↩︎",
    },
    remarkPlugins: [remarkObsidianImages, remarkMath],
    rehypePlugins: [rehypeKatex, rehypeGlueMathPunctuation, rehypeSidenotes],
    shikiConfig: {
      themes: {
        light: "github-light",
        dark: "github-dark",
      },
      wrap: true,
    },
  },
});
