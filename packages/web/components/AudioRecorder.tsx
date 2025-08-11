"use client";

import { cn } from "@/utils";
import { Mic, Square } from "lucide-react";
import { Button } from "./ui/button";
import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";

export default function AudioRecorder({}) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const mediaRecorder = useRef<MediaRecorder | null>(null);
  const chunks = useRef<Blob[]>([]);

  const startRecording = useCallback(async () => {
    // Mock connection logic
    console.log("Connecting to server...");

    try {
      chunks.current = [];
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 44100,
        },
      });

      mediaRecorder.current = new MediaRecorder(stream, {
        mimeType: "audio/webm;codecs=opus",
      });

      mediaRecorder.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunks.current.push(event.data);
        }
      };

      mediaRecorder.current.onstop = async () => {
        const audioBlob = new Blob(chunks.current, { type: "audio/webm" });

        // Stop all tracks
        stream.getTracks().forEach((track) => track.stop());

        // Create the blob URL.
        const blobURL = URL.createObjectURL(audioBlob);
        // Create the `<a download>` element and append it invisibly.
        const a = document.createElement("a");
        a.href = blobURL;
        a.download = "recording.webm";
        a.style.display = "none";
        document.body.append(a);
        a.click();
        // Revoke the blob URL and remove the element.
        setTimeout(() => {
          URL.revokeObjectURL(blobURL);
          a.remove();
          chunks.current = [];
        }, 1000);
      };

      // Record in smaller chunks for more frequent updates (500ms)
      mediaRecorder.current.start(500);
      setIsRecording(true);
      setRecordingTime(0);

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
    } catch (error) {
      toast.error("Failed to access microphone. Please check permissions.");
      console.error("Error starting recording:", error);
      setIsRecording(false);
      setRecordingTime(0);
    }
  }, []);
  const stopRecording = useCallback(async () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (mediaRecorder.current) {
      mediaRecorder.current.stop();
    }
    setIsRecording(false);
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
