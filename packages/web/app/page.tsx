import Chat from "@/components/Chat";
import Header from "@/components/ui/Header";

export default function Home() {
  return (
    <div className="font-sans h-screen flex flex-col">
      <Header />
      <main className="flex-1 p-12">
        <Chat />
      </main>
    </div>
  );
}
