import { Nav } from "@/components/nav";
import { Hero } from "@/components/hero";
import { Providers } from "@/components/providers";
import { CodeExamples } from "@/components/code-examples";
import { Problems } from "@/components/problems";
import { Features } from "@/components/features";
import { ActivityLog } from "@/components/activity-log";
import { VideoSection } from "@/components/video-section";
import { CTA } from "@/components/cta";
import { StarHistory } from "@/components/star-history";
import { Footer } from "@/components/footer";

export default function Home() {
  return (
    <div className="relative min-h-screen bg-noise">
      <Nav />
      <main className="mx-auto max-w-[1200px]">
        <Hero />
        <Providers />
        <CodeExamples />
        <Problems />
        <Features />
        <ActivityLog />
        <VideoSection />
        <StarHistory />
        <CTA />
        <Footer />
      </main>
    </div>
  );
}
