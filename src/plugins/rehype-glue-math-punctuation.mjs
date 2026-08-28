import { visit } from "unist-util-visit";

const LEADING_PUNCT = /^[,.;:!?)\]]+/;

/**
 * Inline KaTeX output is an element, so browsers treat the boundary before a
 * following "," or "." as a line-break opportunity — punctuation after math
 * can start a line. Wrap the math span and its trailing punctuation in a
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
      for (let i = 0; i < kids.length - 1; i++) {
        const el = kids[i];
        const next = kids[i + 1];
        // Display math is excluded structurally: its .katex span is the sole
        // child of a .katex-display wrapper, which carries only that class.
        const isInlineKatex =
          el.type === "element" &&
          Array.isArray(el.properties?.className) &&
          el.properties.className.includes("katex");
        if (!isInlineKatex || next.type !== "text") continue;
        const punct = next.value.match(LEADING_PUNCT)?.[0];
        if (!punct) continue;
        next.value = next.value.slice(punct.length);
        const glued = {
          type: "element",
          tagName: "span",
          properties: { className: ["math-punct"] },
          children: [el, { type: "text", value: punct }],
        };
        kids.splice(i, next.value ? 1 : 2, glued);
      }
    });
  };
}
