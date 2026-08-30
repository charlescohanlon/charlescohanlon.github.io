import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";

// Each post is a folder in the vault containing post.md (plus assets/, notes,
// etc.) — only post.md is published, so working files are safe to keep there.
const posts = defineCollection({
  loader: glob({ pattern: "**/post.{md,mdx}", base: "./content/posts" }),
  schema: ({ image }) =>
    z.object({
      title: z.string(),
      description: z.string().optional(),
      date: z.coerce.date(),
      updated: z.coerce.date().optional(),
      tags: z.array(z.string()).default([]),
      draft: z.boolean().default(false),
      // Decorative banner above the title, resolved relative to post.md.
      // A post can instead ship a Hero.astro next to its post.md for an
      // animated hero (it takes precedence; see blog/[...slug].astro) —
      // `hero` then serves as the static fallback.
      hero: image().optional(),
      // Link-preview card (og:image / twitter:image), resolved relative to
      // post.md. Aim for 1200x630 (or 2x) — a screenshot of the page top
      // works well for posts with a hero.
      ogImage: image().optional(),
    }),
});

// Research projects. Each entry may list where the work was presented
// (posters, talks, papers) under `presentations`.
const research = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./content/research" }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date().optional(),
    image: z.string().optional(),
    link: z.string().url().optional(),
    code: z.string().url().optional(),
    // Full attribution sentence, e.g. "Sponsored by the Menlo Park Fire
    // Protection District" — each entry picks its own verb.
    funding: z.string().optional(),
    presentations: z
      .array(
        z.object({
          title: z.string().optional(),
          authors: z.array(z.string()).optional(),
          venue: z.string(),
          type: z.string().optional(), // e.g. "Poster", "Talk", "Paper"
          year: z.number(),
          pdf: z.string().url().optional(),
        })
      )
      .default([]),
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

// Prose pages: the body is written in Markdown. For list pages (research,
// blog) the body is an optional intro shown above the list.
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
    // GoatCounter endpoint ("https://<code>.goatcounter.com/count");
    // omit to disable analytics
    goatcounter: z.string().url().optional(),
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
  research,
  news,
  pages,
  settings,
};
