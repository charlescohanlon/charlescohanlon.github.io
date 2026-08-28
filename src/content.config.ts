import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

// Each post is a folder in the vault containing post.md (plus assets/, notes,
// etc.) — only post.md is published, so working files are safe to keep there.
const posts = defineCollection({
  loader: glob({ pattern: "**/post.{md,mdx}", base: "./content/posts" }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    date: z.coerce.date(),
    updated: z.coerce.date().optional(),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false),
  }),
});

const publications = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./content/publications" }),
  schema: z.object({
    title: z.string(),
    authors: z.array(z.string()),
    venue: z.string(),
    type: z.string().optional(), // e.g. "Poster", "Talk", "Paper"
    year: z.number(),
    pdf: z.string().url().optional(),
    code: z.string().url().optional(),
    project: z.string().url().optional(),
    bibtex: z.string().optional(),
    selected: z.boolean().default(false),
  }),
});

const portfolio = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./content/portfolio" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date().optional(),
    image: z.string().optional(),
    link: z.string().url().optional(),
    code: z.string().url().optional(),
  }),
});

// News blurbs: the date is frontmatter; the one-line message is the Markdown
// body, so links are written as [text](url) instead of raw HTML.
const news = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./content/news" }),
  schema: z.object({
    date: z.coerce.date(),
  }),
});

// Prose pages: the body is written in Markdown. For list pages (publications,
// portfolio, blog) the body is an optional intro shown above the list.
const pages = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./content/pages" }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
  }),
});

// Site-wide settings, edited from content/settings/site.md.
const settings = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./content/settings" }),
  schema: z.object({
    title: z.string(),
    tagline: z.string(),
    url: z.string().url(),
    author: z.object({
      name: z.string(),
      title: z.string().optional(),
      affiliation: z.string().optional(),
      bio: z.string().optional(),
      photo: z.string().optional(),
      email: z.string().optional(),
      location: z.string().optional(),
    }),
    social: z.object({
      github: z.string().default(""),
      googleScholar: z.string().default(""),
      twitter: z.string().default(""),
      linkedin: z.string().default(""),
      orcid: z.string().default(""),
      pubmed: z.string().default(""),
      bluesky: z.string().default(""),
    }),
    nav: z.array(z.object({ href: z.string(), label: z.string() })),
  }),
});

export const collections = {
  posts,
  publications,
  portfolio,
  news,
  pages,
  settings,
};
