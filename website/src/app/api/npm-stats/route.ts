export const dynamic = "force-dynamic";

const PKG_ENC = "@agentbudget%2fagentbudget";

// Format a Date as YYYY-MM-DD (UTC).
function ymd(d: Date): string {
  return d.toISOString().slice(0, 10);
}

// npm's range endpoint returns at most 18 months of data per request, so we
// page backwards from today to the package's creation date and sum every day.
// The old implementation used `point/last-month`, which only covered a rolling
// 30-day window — that's why the badge showed ~30 days of installs (e.g. 189)
// instead of the all-time total.
export async function GET() {
  try {
    // Discover the package's first-publish date so we know how far back to go.
    let start = new Date("2026-04-01T00:00:00Z"); // safe fallback (first publish)
    try {
      const meta = await fetch(`https://registry.npmjs.org/${PKG_ENC}`, {
        cache: "no-store",
        headers: { "User-Agent": "agentbudget-website/1.0" },
        signal: AbortSignal.timeout(10000),
      });
      const metaJson = await meta.json();
      const created = metaJson?.time?.created;
      if (typeof created === "string") start = new Date(created);
    } catch {
      // keep the fallback start date
    }

    const today = new Date();
    let total = 0;
    let cursorEnd = new Date(today);

    // Page in ~18-month (540-day) windows from today back to `start`.
    while (cursorEnd >= start) {
      const cursorStart = new Date(cursorEnd);
      cursorStart.setUTCDate(cursorStart.getUTCDate() - 539);
      const windowStart = cursorStart < start ? new Date(start) : cursorStart;

      const res = await fetch(
        `https://api.npmjs.org/downloads/range/${ymd(windowStart)}:${ymd(cursorEnd)}/${PKG_ENC}`,
        {
          cache: "no-store",
          headers: { "User-Agent": "agentbudget-website/1.0" },
          signal: AbortSignal.timeout(10000),
        }
      );
      const data = await res.json();
      if (Array.isArray(data?.downloads)) {
        for (const day of data.downloads) {
          if (typeof day?.downloads === "number") total += day.downloads;
        }
      }

      // Move the window to the day before this window's start.
      cursorEnd = new Date(windowStart);
      cursorEnd.setUTCDate(cursorEnd.getUTCDate() - 1);
    }

    return Response.json({ downloads: total || null });
  } catch {
    return Response.json({ downloads: null });
  }
}
