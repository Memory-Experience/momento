import Chat from "@/components/Chat";
import Header from "@/components/ui/Header";

export default function Home() {
  return (
    <div className="font-sans h-screen grid grid-flow-row grid-rows-[62px_1fr]">
      <Header />
      <main className="p-6 min-h-0 overflow-hidden">
        <Chat />
      </main>
    </div>
  );
}
