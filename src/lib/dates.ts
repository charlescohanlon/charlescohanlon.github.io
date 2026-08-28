// Collection dates are parsed from "YYYY-MM-DD" frontmatter as UTC midnight
// (z.coerce.date), so they must also be formatted in UTC — local-timezone
// formatting shows the previous day anywhere west of Greenwich.
export const formatDate = (d: Date, month: "short" | "long" = "short") =>
  d.toLocaleDateString("en-US", {
    year: "numeric",
    month,
    day: "numeric",
    timeZone: "UTC",
  });
