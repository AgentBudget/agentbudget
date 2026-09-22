const REVALIDATE_SECONDS = 60 * 60; // 1 hour

export async function GET() {
  let downloads: number | null = null;

  try {
    // Pepy powers the linked badge/page and provides authoritative total downloads.
    const pepyRes = await fetch(
      "https://api.pepy.tech/api/v2/projects/agentbudget",
      {
        next: { revalidate: REVALIDATE_SECONDS },
        headers: { "User-Agent": "agentbudget-website/1.0" },
        signal: AbortSignal.timeout(10000),
      }
    );

    if (pepyRes.ok) {
      const pepyData = await pepyRes.json();
      const pepyTotal = pepyData?.total_downloads;
      if (typeof pepyTotal === "number" && Number.isFinite(pepyTotal) && pepyTotal >= 0) {
        downloads = pepyTotal;
      }
    }
  } catch (error) {
    console.error("pypi-stats: pepy fetch failed", error);
  }

  // Fallback to pypistats if Pepy is temporarily unavailable.
  if (downloads === null) {
    try {
      const res = await fetch(
        "https://pypistats.org/api/packages/agentbudget/overall",
        {
          next: { revalidate: REVALIDATE_SECONDS },
          headers: { "User-Agent": "agentbudget-website/1.0" },
          signal: AbortSignal.timeout(10000),
        }
      );

      if (res.ok) {
        const data = await res.json();
        let total = 0;
        if (Array.isArray(data?.data)) {
          for (const entry of data.data) {
            if (entry?.category === "with_mirrors" && typeof entry?.downloads === "number") {
              total += entry.downloads;
            }
          }
        }
        downloads = total > 0 ? total : null;
      }
    } catch (error) {
      console.error("pypi-stats: fallback fetch failed", error);
    }
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
