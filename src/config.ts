import { getEntry } from "astro:content";

/**
 * Site-wide settings. All values are edited in Markdown at
 * content/settings/site.md — never in this file.
 */
export async function getSettings() {
  const entry = await getEntry("settings", "site");
  if (!entry) {
    throw new Error(
      "Missing site settings. Create content/settings/site.md."
    );
  }
  return entry.data;
}
