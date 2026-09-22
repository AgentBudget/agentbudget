const PKG_ENC = "@agentbudget%2fagentbudget";
const FIRST_PUBLISH_FALLBACK = "2026-04-01T00:00:00Z";
const REVALIDATE_SECONDS = 60 * 60; // 1 hour

// Format a Date as YYYY-MM-DD (UTC).
function ymd(d: Date): string {
  return d.toISOString().slice(0, 10);
}

export async function GET() {
  let downloads: number | null = null;

  try {
    // Discover package publish date, then ask npm API for a single all-time point total.
    let start = new Date(FIRST_PUBLISH_FALLBACK);
    const meta = await fetch(`https://registry.npmjs.org/${PKG_ENC}`, {
      next: { revalidate: REVALIDATE_SECONDS },
      headers: { "User-Agent": "agentbudget-website/1.0" },
      signal: AbortSignal.timeout(10000),
    });
    if (meta.ok) {
      const metaJson = await meta.json();
      const created = metaJson?.time?.created;
      if (typeof created === "string") {
        const createdDate = new Date(created);
        if (!Number.isNaN(createdDate.getTime())) start = createdDate;
      }
    }

    const pointRes = await fetch(
      `https://api.npmjs.org/downloads/point/${ymd(start)}:${ymd(new Date())}/${PKG_ENC}`,
      {
        next: { revalidate: REVALIDATE_SECONDS },
        headers: { "User-Agent": "agentbudget-website/1.0" },
        signal: AbortSignal.timeout(10000),
      }
    );

    if (pointRes.ok) {
      const data = await pointRes.json();
      if (typeof data?.downloads === "number" && Number.isFinite(data.downloads) && data.downloads >= 0) {
        downloads = data.downloads;
      }
    }
  } catch (error) {
    console.error("npm-stats: fetch failed", error);
  }

  return Response.json(
    { downloads },
    {
      headers: {
        "Cache-Control": `public, max-age=0, s-maxage=${REVALIDATE_SECONDS}, stale-while-revalidate=86400`,
      },
    }
  );
}
