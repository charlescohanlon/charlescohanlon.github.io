// Posts live at content/posts/<folder>/post.md, so entry ids look like
// "my-post-title/post". The public URL drops the trailing "/post".
export const postSlug = (id: string) => id.replace(/\/post$/, "");
