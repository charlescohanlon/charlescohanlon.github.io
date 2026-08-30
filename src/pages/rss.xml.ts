import rss from "@astrojs/rss";
import { getCollection } from "astro:content";
import type { APIContext } from "astro";
import { getSettings } from "@/config";
import { postSlug } from "@/lib/posts";

export async function GET(context: APIContext) {
  const site = await getSettings();
  const posts = (await getCollection("posts", ({ data }) => !data.draft)).sort(
    (a, b) => b.data.date.valueOf() - a.data.date.valueOf()
  );

  return rss({
    title: site.title,
    description: site.tagline,
    site: context.site ?? site.url,
    trailingSlash: false,
    items: posts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.date,
      description: post.data.description ?? "",
      link: `/blog/${postSlug(post.id)}`,
    })),
  });
}
