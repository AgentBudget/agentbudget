export function VideoSection() {
  return (
    <section className="py-20 px-6">
      <div className="text-center mb-8">
        <p className="text-sm font-semibold text-indigo-400 uppercase tracking-widest mb-3">
          See it in action
        </p>
        <h2 className="text-3xl font-bold text-white mb-3">
          Watch a 6-min video all about AgentBudget
        </h2>
        <p className="text-zinc-400 text-base max-w-xl mx-auto">
          A complete walkthrough — from installation to real-time cost enforcement across your AI agents.
        </p>
      </div>
      <div className="mx-auto max-w-3xl rounded-2xl overflow-hidden border border-white/10 shadow-2xl shadow-indigo-950/40">
        <div className="relative w-full" style={{ paddingBottom: "56.25%" }}>
          <iframe
            className="absolute inset-0 w-full h-full"
            src="https://www.youtube.com/embed/lyjbHbyDMqo?si=b2oJk0Kvsf_fMkMH"
            title="AgentBudget — 6-minute overview"
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
            referrerPolicy="strict-origin-when-cross-origin"
            allowFullScreen
          />
        </div>
      </div>
    </section>
  );
}
