import type { Metadata } from "next";
import { GeistSans } from "geist/font/sans";
import { GeistMono } from "geist/font/mono";
import "./globals.css";
import { Nav } from "@/components/Nav";
import { cn } from "@/utils";
import { Toaster } from "@/components/ui/sonner";
import { ThemeProvider } from "@/components/ThemeProvider";

export const metadata: Metadata = {
  title: "Web",
  description:
    "A Next.js application providing a modern web interface for real-time audio recording and transcription. Built with TypeScript, Tailwind CSS, and WebRTC audio capture.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // NOTE: ThemeProvider requires suppressHydrationWarning to suppress hydration warning
  // as the theme differs between server and client rendering due to systems preference.
  // See the note at the bottom of https://github.com/pacocoursey/next-themes?tab=readme-ov-file#with-app
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={cn(
          GeistSans.variable,
          GeistMono.variable,
          "flex flex-col min-h-screen",
        )}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="light"
          enableSystem
          disableTransitionOnChange
        >
          <Nav />
          {children}
          <Toaster position="top-center" richColors={true} />
        </ThemeProvider>
      </body>
    </html>
  );
}
