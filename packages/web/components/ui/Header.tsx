"use client";

import Image from "next/image";
import Link from "next/link";
import { FC } from "react";
import { Button, Sheet } from "@mui/joy";
import { DarkMode, LightMode } from "@mui/icons-material";
import { useTheme } from "next-themes";

const Header: FC = () => {
  const { theme, setTheme } = useTheme();

  const toggleTheme = () => {
    setTheme(theme === "dark" ? "light" : "dark");
  };

  return (
    <Sheet
      component="header"
      variant="outlined"
      sx={{
        px: { xs: 2, md: 6 },
        py: 2,
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        position: "sticky",
        top: 0,
        zIndex: 50,
        borderTop: "none",
        borderLeft: "none",
        borderRight: "none",
        borderBottom: "1px solid",
        borderBottomColor: "divider",
        boxShadow: "sm",
      }}
    >
      <Link href="/" className="flex items-center">
        <Image src="/logo.png" alt="Momento Logo" width={122} height={30} />
      </Link>

      <Button
        onClick={toggleTheme}
        variant="outlined"
        size="sm"
        sx={{
          borderRadius: "xl",
          gap: 1,
        }}
        startDecorator={
          theme === "dark" ? (
            <LightMode fontSize="small" />
          ) : (
            <DarkMode fontSize="small" />
          )
        }
      >
        {theme === "dark" ? "Light" : "Dark"} Mode
      </Button>
    </Sheet>
  );
};

export default Header;
