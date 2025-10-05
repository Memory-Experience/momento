"use client";

import { useEffect, useState } from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";
import { CssVarsProvider } from "@mui/joy/styles";
import { extendTheme } from "@mui/joy/styles";

// Create a custom JoyUI theme that works with light/dark modes
const theme = extendTheme({
  cssVarPrefix: "joy",
  colorSchemes: {
    light: {
      palette: {
        primary: {
          50: "#f0f9ff",
          100: "#e0f2fe",
          200: "#bae6fd",
          300: "#7dd3fc",
          400: "#38bdf8",
          500: "#0ea5e9",
          600: "#0284c7",
          700: "#0369a1",
          800: "#075985",
          900: "#0c4a6e",
        },
        background: {
          body: "#ffffff",
          surface: "#ffffff",
          level1: "#f8fafc",
          level2: "#f1f5f9",
          level3: "#e2e8f0",
        },
        text: {
          primary: "#171717",
          secondary: "#64748b",
          tertiary: "#94a3b8",
        },
      },
    },
    dark: {
      palette: {
        primary: {
          50: "#0c4a6e",
          100: "#075985",
          200: "#0369a1",
          300: "#0284c7",
          400: "#0ea5e9",
          500: "#38bdf8",
          600: "#7dd3fc",
          700: "#bae6fd",
          800: "#e0f2fe",
          900: "#f0f9ff",
        },
        background: {
          body: "#0a0a0a",
          surface: "#0a0a0a",
          level1: "#1a1a1a",
          level2: "#262626",
          level3: "#404040",
        },
        text: {
          primary: "#ededed",
          secondary: "#a1a1aa",
          tertiary: "#71717a",
        },
      },
    },
  },
});

// Component to sync next-themes with JoyUI theme
function ThemeSync() {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!mounted) return;

    const updateJoyTheme = () => {
      const isDark = document.documentElement.classList.contains("dark");
      document.documentElement.setAttribute(
        "data-joy-color-scheme",
        isDark ? "dark" : "light",
      );
    };

    // Initial sync
    updateJoyTheme();

    // Watch for theme changes
    const observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (
          mutation.type === "attributes" &&
          mutation.attributeName === "class"
        ) {
          updateJoyTheme();
        }
      });
    });

    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

    return () => observer.disconnect();
  }, [mounted]);

  return null;
}

export function ThemeProvider({
  children,
  ...props
}: React.ComponentProps<typeof NextThemesProvider>) {
  return (
    <NextThemesProvider {...props}>
      <CssVarsProvider theme={theme} defaultMode="system">
        <ThemeSync />
        {children}
      </CssVarsProvider>
    </NextThemesProvider>
  );
}
