import { visit } from "unist-util-visit";

const PREFIX_RUN = /\S+$/; // e.g. "(", "top-" — glued to the math that follows
const SUFFIX_RUN = /^\S+/; // e.g. ",", ")", "-th" — glued to the math before

/**
 * Inline KaTeX output is an element, so browsers treat its boundaries as
 * line-break opportunities regardless of the adjacent characters: an opening
 * "(" can be stranded at the end of a line before math, and punctuation after
 * math can start a line. Wrap the math span together with any directly
 * adjacent non-whitespace runs ("(", ")", ",", "top-", "-th", …) in a
 * `white-space: nowrap` span (`.math-punct`) so they stay on one line.
 */
export default function rehypeGlueMathPunctuation() {
  return (tree) => {
    visit(tree, "element", (node) => {
      // Don't descend into spans this plugin created (re-wrapping the katex +
      // punctuation pair inside them recurses forever) or into KaTeX output —
      // glue candidates are only ever siblings of a katex span, and its huge,
      // deeply nested subtrees dominate both walk time and recursion depth.
      const cls = node.properties?.className;
      if (
        Array.isArray(cls) &&
        (cls.includes("math-punct") || cls.some((c) => c.startsWith("katex")))
      ) {
        return "skip";
      }
      const kids = node.children;
      if (!kids) return;
      for (let i = 0; i < kids.length; i++) {
        const el = kids[i];
        // Display math is excluded structurally: its .katex span is the sole
        // child of a .katex-display wrapper, which carries only that class.
        const isInlineKatex =
          el.type === "element" &&
          Array.isArray(el.properties?.className) &&
          el.properties.className.includes("katex");
        if (!isInlineKatex) continue;

        const prev = i > 0 && kids[i - 1].type === "text" ? kids[i - 1] : null;
        const next =
          i < kids.length - 1 && kids[i + 1].type === "text" ? kids[i + 1] : null;
        const prefix = prev?.value.match(PREFIX_RUN)?.[0];
        const suffix = next?.value.match(SUFFIX_RUN)?.[0];
        if (!prefix && !suffix) continue;

        const glued = {
          type: "element",
          tagName: "span",
          properties: { className: ["math-punct"] },
          children: [
            ...(prefix ? [{ type: "text", value: prefix }] : []),
            el,
            ...(suffix ? [{ type: "text", value: suffix }] : []),
          ],
        };
        // Trim the runs off their source text nodes; drop a node its run
        // consumed entirely, and splice the glued span in over the range.
        let start = i;
        let remove = 1;
        if (prefix) {
          prev.value = prev.value.slice(0, -prefix.length);
          if (!prev.value) {
            start -= 1;
            remove += 1;
          }
        }
        if (suffix) {
          next.value = next.value.slice(suffix.length);
          if (!next.value) remove += 1;
        }
        kids.splice(start, remove, glued);
        i = start; // continue after the glued span
      }
    });
  };
}
