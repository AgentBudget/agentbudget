const REVALIDATE_SECONDS = 60 * 60; // 1 hour
const WINDOW_DAYS = 14;

// GitHub traffic API — requires a token with `repo` or `public_repo` read access.
// Set GITHUB_TOKEN in your environment (Vercel env vars, .env.local, etc.)
// Returns clone events for the last 14 days as a proxy for Go module installs.
// Note: there is no official Go module download counter — proxy.golang.org does
// not expose counts, and pkg.go.dev only shows an unquantified "Used by" count.

export async function GET() {
  const token = process.env.GITHUB_TOKEN;
  const headers = {
    "Cache-Control": `public, max-age=0, s-maxage=${REVALIDATE_SECONDS}, stale-while-revalidate=86400`,
  };

  if (!token) {
    return Response.json({ clones: null, is_proxy: true, window_days: WINDOW_DAYS }, { headers });
  }

  try {
    const res = await fetch("https://api.github.com/repos/AgentBudget/agentbudget/traffic/clones", {
      next: { revalidate: REVALIDATE_SECONDS },
      headers: {
        Authorization: ["Bearer", token].join(" "),
        Accept: "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "agentbudget-website/1.0",
      },
      signal: AbortSignal.timeout(10000),
    });

    if (!res.ok) {
      console.error("go-stats: GitHub API error", res.status);
      return Response.json({ clones: null, is_proxy: true, window_days: WINDOW_DAYS }, { headers });
    }

    const data = await res.json();
    const uniques: number = typeof data?.uniques === "number" ? data.uniques : 0;
    const count: number = typeof data?.count === "number" ? data.count : 0;

    return Response.json(
      { clones: count, unique_cloners: uniques, is_proxy: true, window_days: WINDOW_DAYS },
      { headers }
    );
  } catch (error) {
    console.error("go-stats: fetch failed", error);
    return Response.json({ clones: null, is_proxy: true, window_days: WINDOW_DAYS }, { headers });
  }
}
