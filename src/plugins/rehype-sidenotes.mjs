/**
 * Rehype plugin that turns GFM footnotes into Tufte-style sidenotes.
 *
 * After each footnote reference (<sup><a data-footnote-ref>) it inserts
 *   <span class="sidenote" data-note="N">…note content…</span>
 * carrying a copy of the note, and leaves the original reference and the
 * trailing footnotes section in place. CSS then shows one presentation per
 * viewport: on wide screens the span floats into the right margin beside
 * its reference and the bottom section is hidden; on narrow screens the
 * span is hidden and the reference links down to the bottom section.
 */
import { visit } from "unist-util-visit";

const isElement = (node, tagName) =>
  node?.type === "element" && node.tagName === tagName;

// A footnote definition <li> holds block content (usually one <p>, possibly
// with display-math blocks) ending with a back-reference arrow. Flatten it to
// inline nodes so it can live inside the referencing paragraph, joining
// consecutive paragraphs with <br>. Display math is already a
// <span class="katex-display"> (rehype-katex runs first), which is inline-valid
// and breaks its own line via CSS, so it's carried over as-is.
function inlineContent(li) {
  const out = [];
  let prevFlowsInline = false;
  for (const block of li.children) {
    if (isElement(block, "p")) {
      const kept = block.children.filter(
        (c) => !(c.type === "element" && c.properties?.dataFootnoteBackref !== undefined),
      );
      // drop whitespace left behind before the back-reference
      while (kept.length) {
        const last = kept[kept.length - 1];
        if (last.type === "text" && !last.value.trim()) kept.pop();
        else break;
      }
      if (prevFlowsInline) out.push({ type: "element", tagName: "br", properties: {}, children: [] });
      out.push(...kept);
      prevFlowsInline = true;
    } else if (
      block.type === "element" &&
      (block.properties?.className ?? []).includes("katex-display")
    ) {
      out.push(block);
      prevFlowsInline = false;
    }
  }
  return out;
}

export default function rehypeSidenotes() {
  return (tree) => {
    // Collect footnote definitions from the trailing section.
    const defs = new Map();
    visit(tree, { tagName: "section" }, (node) => {
      if (node.properties?.dataFootnotes === undefined) return;
      visit(node, { tagName: "li" }, (li) => {
        if (li.properties?.id) defs.set(li.properties.id, inlineContent(li));
      });
    });
    if (defs.size === 0) return;

    // Insert a sidenote span after each reference, numbered to match the
    // reference's own text so margin and bottom numbering always agree.
    visit(tree, { tagName: "sup" }, (node, index, parent) => {
      const ref = node.children?.[0];
      if (!isElement(ref, "a") || ref.properties?.dataFootnoteRef === undefined) {
        return;
      }
      const content = defs.get(ref.properties.href?.slice(1));
      if (!content) return;

      // Mark the reference so page CSS can target it without :has()
      node.properties.className = ["footnote-ref"];
      const number = ref.children?.[0]?.value ?? "";
      parent.children.splice(index + 1, 0, {
        type: "element",
        tagName: "span",
        properties: { className: ["sidenote"], dataNote: number },
        children: structuredClone(content),
      });
      return index + 2;
    });
  };
}
