import { cn } from "@/utils";
import { FC } from "react";

interface ConnectionIndicatorProps {
  isConnected: boolean;
  isChecking?: boolean;
  className?: string;
}

const ConnectionIndicator: FC<ConnectionIndicatorProps> = ({
  isConnected,
  isChecking = false,
  className,
}) => {
  return (
    <div
      className={cn(
        "fixed bottom-4 right-4 z-50",
        "p-3 bg-card border border-border/50 rounded-full",
        "flex items-center gap-2",
        "animate-in fade-in slide-in-from-bottom-4 duration-300",
        className,
      )}
    >
      <div
        className={cn(
          "w-2 h-2 rounded-full transition-colors duration-300",
          isChecking
            ? "bg-yellow-500 animate-pulse"
            : isConnected
              ? "bg-green-500"
              : "bg-red-500",
        )}
      />
      <span className="text-sm text-muted-foreground">
        {isChecking
          ? "Checking..."
          : isConnected
            ? "Connected"
            : "Disconnected"}
      </span>
    </div>
  );
};

export default ConnectionIndicator;
