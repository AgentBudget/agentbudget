export const dynamic = "force-dynamic";

export async function GET() {
  try {
    // npm public downloads API — last 30 days, no auth required
    const res = await fetch(
      "https://api.npmjs.org/downloads/point/last-month/agentbudget",
      {
        cache: "no-store",
        headers: { "User-Agent": "agentbudget-website/1.0" },
        signal: AbortSignal.timeout(10000),
      }
    );
    const data = await res.json();
    const downloads = typeof data?.downloads === "number" ? data.downloads : null;
    return Response.json({ downloads });
  } catch {
    return Response.json({ downloads: null });
  }
}
