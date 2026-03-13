export const dynamic = "force-dynamic"; // always fetch fresh

export async function GET() {
  try {
    const res = await fetch(
      "https://pypistats.org/api/packages/agentbudget/overall",
      {
        cache: "no-store",
        headers: { "User-Agent": "agentbudget-website/1.0" },
        signal: AbortSignal.timeout(10000),
      }
    );
    const data = await res.json();
    // Sum "with_mirrors" to match pepy.tech total
    let total = 0;
    if (Array.isArray(data?.data)) {
      for (const entry of data.data) {
        if (entry.category === "with_mirrors") {
          total += entry.downloads;
        }
      }
    }
    return Response.json({ downloads: total || null });
  } catch {
    return Response.json({ downloads: null });
  }
}
