/**
 * Remark plugin that converts Obsidian-style image embeds (![[file.png]])
 * into standard Markdown image nodes (![alt](./file.png)), then wraps
 * image-led paragraphs in <figure>, with any trailing text (the caption
 * line written directly under the embed) as its <figcaption>.
 *
 * This lets you write in Obsidian with its native embed syntax and have the
 * images render in Astro without changing the source markdown.
 */
import { visit } from "unist-util-visit";

// A caption's leading soft break / whitespace from the source line break.
function trimLeadingWhitespace(nodes) {
  const out = [...nodes];
  while (out.length && (out[0].type === "break" || (out[0].type === "text" && !out[0].value.trim()))) {
    out.shift();
  }
  if (out.length && out[0].type === "text") {
    out[0] = { ...out[0], value: out[0].value.replace(/^\s+/, "") };
  }
  return out;
}

// Obsidian may append an alias/size after the filename (![[img.png|400]]).
const OBSIDIAN_EMBED_RE =
  /!\[\[([^\]|]+\.(?:png|jpe?g|gif|svg|webp|avif))(?:\|[^\]]*)?\]\]/gi;

export default function remarkObsidianImages() {
  return (tree) => {
    visit(tree, "paragraph", (node) => {
      const newChildren = [];
      let changed = false;

      for (const child of node.children) {
        if (child.type !== "text") {
          newChildren.push(child);
          continue;
        }

        let lastIndex = 0;
        let match;
        let childChanged = false;
        OBSIDIAN_EMBED_RE.lastIndex = 0;

        while ((match = OBSIDIAN_EMBED_RE.exec(child.value)) !== null) {
          childChanged = true;
          // text before the embed
          if (match.index > lastIndex) {
            newChildren.push({
              type: "text",
              value: child.value.slice(lastIndex, match.index),
            });
          }
          const filename = match[1];
          // Bare embeds refer to images in the post folder's assets/ dir;
          // embeds that already carry a path are used as-is.
          const url = filename.includes("/")
            ? `./${filename}`
            : `./assets/${filename}`;
          newChildren.push({
            type: "image",
            url,
            alt: filename.replace(/^.*\//, "").replace(/\.[^.]+$/, ""),
          });
          lastIndex = OBSIDIAN_EMBED_RE.lastIndex;
        }

        if (!childChanged) {
          // no matches — keep the original child as-is
          newChildren.push(child);
          continue;
        }

        changed = true;
        // remaining text after the last match
        if (lastIndex < child.value.length) {
          newChildren.push({
            type: "text",
            value: child.value.slice(lastIndex),
          });
        }
      }

      if (changed) {
        node.children = newChildren;
      }
    });

    // Second pass: any paragraph that starts with an image becomes a figure,
    // with the rest of the paragraph (if any) as its caption.
    visit(tree, "paragraph", (node) => {
      const [first, ...rest] = node.children;
      if (!first || first.type !== "image" || node.data?.hName) return;

      const caption = trimLeadingWhitespace(rest);
      node.data = { hName: "figure" };
      node.children = caption.length
        ? [first, { type: "strong", data: { hName: "figcaption" }, children: caption }]
        : [first];
    });
  };
}
