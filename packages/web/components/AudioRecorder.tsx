"use client";

import { cn } from "@/utils";
import { MessageCircleQuestionMark, Mic, Square } from "lucide-react";
import { Button } from "./ui/button";
import { useCallback, useContext, useRef, useState } from "react";
import { toast } from "sonner";
import { ChatContext } from "@/context/ChatContext";

export default function AudioRecorder({}) {
  const { setMode, isRecording, setIsRecording, setTranscriptions } =
    useContext(ChatContext);
  const [recordingTime, setRecordingTime] = useState(0);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const chunks = useRef<Blob[]>([]);
  const sessionId = useRef<string | null>(null);
  const eventSource = useRef<EventSource | null>(null);

  const sendData = useCallback(async (pcmBytes: Uint8Array) => {
    if (!sessionId.current) {
      console.error("No session ID available");
      return;
    }

    // Skip sending very small chunks (likely silent)
    if (pcmBytes.byteLength < 10) {
      return;
    }

    // Create a proper copy to avoid any potential shared buffer issues
    const audioBuffer = pcmBytes.slice(0);

    // // Send the audio chunk
    // socketRef.current.sendMessage('audio', audioChunk);
    try {
      await fetch("/api/transcribe", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          sessionId: sessionId.current,
          audioData: Array.from(audioBuffer),
        }),
      });
    } catch (err) {
      console.error("Error in audio chunk callback:", err);
    }
  }, []);

  const startRecording = useCallback(async () => {
    console.log("Connecting to server...");

    try {
      chunks.current = [];
      setTranscriptions([]);

      // Generate session ID
      sessionId.current = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      // Start Server-Sent Events connection
      eventSource.current = new EventSource(
        `/api/transcribe?sessionId=${sessionId.current}`,
      );

      eventSource.current.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          if (data.type === "connected") {
            console.log("Connected to transcription service");
            toast.success("Connected to transcription service");
          } else if (data.type === "transcript") {
            setTranscriptions((prev) => [...prev, data.text]);
          }
        } catch (error) {
          console.error("Error parsing SSE data:", error);
        }
      };

      eventSource.current.onerror = (error) => {
        console.error("SSE connection error:", error);
        toast.error("Connection error occurred");
      };

      // Request microphone with explicit constraints
      const constraints = {
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
        video: false,
      };
      const stream = await navigator.mediaDevices.getUserMedia(constraints);

      // Create AudioContext WITHOUT specifying sample rate - let it match the native rate
      const audioCtx = new AudioContext({
        latencyHint: "interactive",
      });
      audioCtxRef.current = audioCtx;

      console.debug(
        `Created AudioContext with sample rate: ${audioCtx.sampleRate}Hz and ${audioCtx}`,
      );

      if (audioCtx.state === "suspended") {
        await audioCtx.resume();
      }

      // Create MediaStream source
      const source = audioCtx.createMediaStreamSource(stream);

      try {
        await audioCtx.audioWorklet.addModule(
          "/worklets/audio-recorder.worklet.js",
        );
      } catch (workletError) {
        console.error("Error loading audio worklet:", workletError);
        toast.error(
          "Failed to load audio processing module. Please try again.",
        );
        return;
      }

      // Create recorder node with detailed parameters - now the worklet will handle resampling
      const recorderNode = new AudioWorkletNode(
        audioCtx,
        "audio-recorder-processor",
        {
          numberOfInputs: 1,
          numberOfOutputs: 1,
          channelCount: 1,
          processorOptions: {
            nativeContextSampleRate: audioCtx.sampleRate,
            targetSampleRate: 16000, // We still want 16kHz output for consistency
          },
        },
      );

      // Enhanced message handler
      recorderNode.port.onmessage = async (event: MessageEvent) => {
        // Check if it's a debug message
        if (event.data && event.data.type === "debug") {
          return;
        }

        // Check if it's an audio level message
        if (event.data && event.data.type === "level") {
          // This is where you would handle audio level updates
          // setAudioLevel(event.data.value);
          return;
        }

        // Otherwise it's audio data
        const pcmBytes = event.data;
        if (pcmBytes instanceof Uint8Array && pcmBytes.byteLength > 0) {
          const byteCopy = new Uint8Array(pcmBytes);
          sendData(byteCopy);
        }
      };

      // Configure worklet
      recorderNode.port.postMessage({
        command: "init",
        config: {
          bufferDuration: 100, // 100ms chunks
          nativeSampleRate: audioCtx.sampleRate,
          targetSampleRate: 16000,
        },
      });

      // Connect nodes
      source.connect(recorderNode);

      const silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;
      recorderNode.connect(silentGain).connect(audioCtx.destination);

      setMode("memory");
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
  }, [sendData, setMode, setIsRecording, setTranscriptions]);

  const askQuestion = useCallback(async () => {
    setMode("question");
  }, [setMode]);

  const stopRecording = useCallback(async () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (audioCtxRef.current) {
      audioCtxRef.current
        .close()
        .catch((err) => console.error("Error closing AudioContext", err));
      audioCtxRef.current = null;
    }
    if (eventSource.current) {
      eventSource.current.close();
      eventSource.current = null;
    }

    // Notify server to end the session
    if (sessionId.current) {
      try {
        await fetch(`/api/transcribe?sessionId=${sessionId.current}`, {
          method: "DELETE",
        });
      } catch (error) {
        console.error("Error ending session:", error);
      }
    }

    setIsRecording(false);
    setRecordingTime(0);
    sessionId.current = null;
  }, [setIsRecording]);

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
  };

  return (
    <div
      className={cn(
        "w-full p-4 pb-6 flex flex-col items-center justify-center",
        "bg-gradient-to-t from-card via-card/90 to-card/0",
      )}
    >
      {!isRecording ? (
        <div className="p-4 bg-card flex flex-row items-center gap-2">
          <Button onClick={startRecording} className="rounded-full" size="lg">
            <Mic className="mr-2 h-4 w-4" />
            Record a Memory
          </Button>
          <Button onClick={askQuestion} className="rounded-full" size="lg">
            <MessageCircleQuestionMark className="mr-2 h-4 w-4" />
            Ask a Question
          </Button>
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
