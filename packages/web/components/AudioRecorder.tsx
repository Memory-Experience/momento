"use client";

import { cn } from "@/utils";
import { Mic, Square } from "lucide-react";
import { Button } from "./ui/button";
import { useCallback, useRef, useState } from "react";

export default function AudioRecorder({}) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const startRecording = useCallback(async () => {
    // Mock connection logic
    console.log("Connecting to server...");
    setIsRecording(true);
    setRecordingTime(0);

    // Start timer
    timerRef.current = setInterval(() => {
      setRecordingTime((prev) => prev + 1);
    }, 1000);
  }, []);
  const stopRecording = useCallback(async () => {
    setIsRecording(false);
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setRecordingTime(0);
  }, []);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div
      className={cn(
        "fixed bottom-0 left-0 w-full p-4 pb-6 flex items-center justify-center",
        "bg-gradient-to-t from-card via-card/90 to-card/0",
      )}
    >
      {!isRecording ? (
        <div className="p-4 bg-card flex flex-col items-center gap-2">
          <Button onClick={startRecording} className="rounded-full" size="lg">
            <Mic className="mr-2 h-4 w-4" />
            Connect
          </Button>
          <p className="text-xs text-muted-foreground">
            Click to start recording
          </p>
        </div>
      ) : (
        <div className="p-4 bg-card border border-border/50 rounded-full flex items-center gap-4 max-w-4xl">
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
            onClick={stopRecording}
            variant="destructive"
            className="rounded-full"
          >
            <Square className="mr-2 h-4 w-4" />
            Stop Recording
          </Button>
        </div>
      )}
    </div>
  );
}
