import { Square } from "lucide-react";
import { Button } from "./ui/button";
import { useState, useEffect } from "react";
import { cn } from "@/utils";

interface RecordingIndicatorProps {
  isRecording: boolean;
  onStopRecording: () => void;
}

export default function RecordingIndicator({
  isRecording,
  onStopRecording,
}: RecordingIndicatorProps) {
  const [recordingTime, setRecordingTime] = useState(0);

  useEffect(() => {
    let interval: NodeJS.Timeout | null = null;

    if (isRecording) {
      setRecordingTime(0);
      interval = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } else if (interval) {
      clearInterval(interval);
      setRecordingTime(0);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isRecording]);

  if (!isRecording) return null;

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div
      className={cn(
        "p-4 bg-card border border-border/50 rounded-full",
        "flex items-center gap-4 max-w-md mx-auto",
        "animate-in fade-in slide-in-from-bottom-4 duration-300",
      )}
    >
      <div className="flex items-center gap-2">
        <div className="text-lg font-mono font-bold text-red-500">
          {formatTime(recordingTime)}
        </div>
        <div className="flex items-center gap-1">
          <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
          <span className="text-sm text-muted-foreground">Recording</span>
        </div>
      </div>

      <Button
        onClick={onStopRecording}
        variant="destructive"
        className="rounded-full"
      >
        <Square className="mr-2 h-4 w-4" />
        Stop Recording
      </Button>
    </div>
  );
}
