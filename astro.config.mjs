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
  redirects: {
    "/portfolio": "/research",
    "/publications": "/research",
  },
  integrations: [mdx(), sitemap()],
  vite: {
    plugins: [tailwindcss()],
  },
  markdown: {
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
