import React from "react";
import { cn } from "@/utils/index";

export const Card = ({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) => (
  <div
    className={cn(
      "rounded-lg border bg-card text-card-foreground shadow-sm p-4",
      className,
    )}
    {...props}
  />
);
