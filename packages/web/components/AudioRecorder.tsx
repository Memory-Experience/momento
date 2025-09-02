"use client";

import { cn } from "@/utils";
import { MessageCircleQuestionMark, Mic, Square } from "lucide-react";
import { Button } from "./ui/button";
import { useCallback, useContext, useRef, useState, useEffect } from "react";
import { toast } from "sonner";
import { ChatContext, TranscriptionItem } from "@/context/ChatContext";

export default function AudioRecorder({}) {
  const { mode, setMode, isRecording, setIsRecording, setTranscriptions } =
    useContext(ChatContext);
  const [recordingTime, setRecordingTime] = useState(0);
  const [waitingForAnswer, setWaitingForAnswer] = useState(false);
  const waitingForAnswerRef = useRef(false);
  const hasReceivedAnswerRef = useRef(false);
  // Track whether the backend has confirmed the SSE/gRPC connection
  const connectionConfirmedRef = useRef(false);
  // Store connection timeout id so it can be cleared from any handler
  const connectionTimeoutRef = useRef<number | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const recorderNodeRef = useRef<AudioWorkletNode | null>(null);
  const chunks = useRef<Blob[]>([]);
  const sessionId = useRef<string | null>(null);
  const eventSource = useRef<EventSource | null>(null);

  const sendData = useCallback(async (pcmBytes: Uint8Array) => {
    if (!sessionId.current || pcmBytes.byteLength < 10) return;
    // Create a proper copy to avoid any potential shared buffer issues
    const audioBuffer = pcmBytes.slice(0);

    try {
      await fetch("/api/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId: sessionId.current,
          audioData: Array.from(audioBuffer),
        }),
      });
    } catch (err) {
      console.error("Error sending audio data:", err);
    }
  }, []);

  const recordAudio = useCallback(
    async (type: "memory" | "question") => {
      try {
        chunks.current = [];
        setTranscriptions([]);
        waitingForAnswerRef.current = false;
        hasReceivedAnswerRef.current = false;
        connectionConfirmedRef.current = false;

        // Generate session ID
        sessionId.current = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

        // Start Server-Sent Events connection
        eventSource.current = new EventSource(
          `/api/transcribe?sessionId=${sessionId.current}&type=${type}`,
        );

        let connectionConfirmed = false;

        // Set a timeout to detect if we never get a "connected" message
        connectionTimeoutRef.current = window.setTimeout(() => {
          if (!connectionConfirmed) {
            console.error(
              "Connection timeout - no confirmation received from backend",
            );
            toast.error(
              "Backend service is not responding. Please check if it's running.",
            );
            if (eventSource.current) {
              eventSource.current.close();
              eventSource.current = null;
            }
            setMode(undefined);
            setIsRecording(false);
            sessionId.current = null;
          }
        }, 10000); // 10 second timeout

        eventSource.current.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);

            if (data.type === "connected") {
              connectionConfirmed = true;
              if (connectionTimeoutRef.current !== null) {
                clearTimeout(connectionTimeoutRef.current);
                connectionTimeoutRef.current = null;
              }
              console.log("Connected to transcription service");
              // Show success toast only if this is the first connection confirmation
              if (!connectionConfirmedRef.current) {
                connectionConfirmedRef.current = true;
                toast.success("Connected to transcription service");
              }
            } else if (data.type === "error") {
              if (connectionTimeoutRef.current !== null) {
                clearTimeout(connectionTimeoutRef.current);
                connectionTimeoutRef.current = null;
              }
              console.error("Backend error:", data.message);
              toast.error(data.message || "Unknown error from backend");
              // Clean up on backend error
              if (eventSource.current) {
                eventSource.current.close();
                eventSource.current = null;
              }
              setMode(undefined);
              setIsRecording(false);
              sessionId.current = null;
            } else if (["transcript", "answer", "memory"].includes(data.type)) {
              // Validate required fields exist before creating the transcription item
              if (data.text !== undefined) {
                const transcriptionItem: TranscriptionItem = {
                  type: data.type,
                  text: data.text,
                  timestamp: data.timestamp || new Date().toISOString(),
                };
                setTranscriptions((prev) => [...prev, transcriptionItem]);

                if (data.type === "answer" && !hasReceivedAnswerRef.current) {
                  hasReceivedAnswerRef.current = true;
                  toast.success("Receiving answer to your question!");
                }
              } else {
                console.warn(
                  `Received ${data.type} message without text content:`,
                  data,
                );
              }
            } else {
              console.warn("Received unknown message type:", data.type, data);
            }
          } catch (error) {
            console.error("Error parsing SSE data:", error);
          }
        };

        eventSource.current.onerror = (error) => {
          console.error("SSE connection error:", error);

          // Clear the connection timeout on error
          if (connectionTimeoutRef.current !== null) {
            clearTimeout(connectionTimeoutRef.current);
            connectionTimeoutRef.current = null;
          }

          if (eventSource.current?.readyState === EventSource.CLOSED) {
            // Natural connection close - clean up
            if (type === "question" && waitingForAnswerRef.current) {
              setWaitingForAnswer(false);
              waitingForAnswerRef.current = false;
              setMode(undefined);
              sessionId.current = null;
              if (hasReceivedAnswerRef.current) {
                toast.success("Answer completed!");
              }
            } else if (type === "memory") {
              sessionId.current = null;
              eventSource.current = null;
            }
          } else if (
            eventSource.current?.readyState === EventSource.CONNECTING
          ) {
            // Connection failed - show error
            toast.error(
              "Failed to connect to backend service. Please check if it's running.",
            );
            if (eventSource.current) {
              eventSource.current.close();
              eventSource.current = null;
            }
            setMode(undefined);
            setIsRecording(false);
            sessionId.current = null;
          }
        };

        // Request microphone and setup audio processing
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: true },
        });

        const audioCtx = new AudioContext({ latencyHint: "interactive" });
        audioCtxRef.current = audioCtx;

        if (audioCtx.state === "suspended") await audioCtx.resume();

        const source = audioCtx.createMediaStreamSource(stream);
        await audioCtx.audioWorklet.addModule(
          "/worklets/audio-recorder.worklet.js",
        );

        const recorderNode = new AudioWorkletNode(
          audioCtx,
          "audio-recorder-processor",
          {
            numberOfInputs: 1,
            numberOfOutputs: 1,
            channelCount: 1,
            processorOptions: {
              nativeContextSampleRate: audioCtx.sampleRate,
              targetSampleRate: 16000,
            },
          },
        );

        recorderNodeRef.current = recorderNode;

        recorderNode.port.onmessage = (event: MessageEvent) => {
          if (event.data?.type) return; // Skip debug/level messages
          const pcmBytes = event.data;
          if (pcmBytes instanceof Uint8Array && pcmBytes.byteLength > 0) {
            sendData(new Uint8Array(pcmBytes));
          }
        };

        recorderNode.port.postMessage({
          command: "init",
          config: {
            bufferDuration: 100,
            nativeSampleRate: audioCtx.sampleRate,
            targetSampleRate: 16000,
          },
        });

        source.connect(recorderNode);
        const silentGain = audioCtx.createGain();
        silentGain.gain.value = 0;
        recorderNode.connect(silentGain).connect(audioCtx.destination);

        setMode(type);
        setIsRecording(true);
        setRecordingTime(0);
        timerRef.current = setInterval(
          () => setRecordingTime((prev) => prev + 1),
          1000,
        );
      } catch (error) {
        // Clear the connection timeout on error
        if (connectionTimeoutRef.current !== null) {
          clearTimeout(connectionTimeoutRef.current);
          connectionTimeoutRef.current = null;
        }

        setMode(undefined);
        setIsRecording(false);
        setRecordingTime(0);

        if (error instanceof Error) {
          if (error.name === "NotAllowedError") {
            toast.error(
              "Microphone access denied. Please allow microphone permissions.",
            );
          } else if (error.message.includes("Failed to fetch")) {
            toast.error(
              "Connection failed. Please check if the backend service is running.",
            );
          } else {
            toast.error("An error occurred. Please try again.");
          }
        } else {
          toast.error("An unexpected error occurred.");
        }
        console.error("Recording failed:", error);
      }
    },
    [sendData, setMode, setIsRecording, setTranscriptions],
  );

  const startRecording = useCallback(async () => {
    try {
      await recordAudio("memory");
    } catch (error) {
      // Error already handled in recordAudio function
      console.error("Unable to start recording memory:", error);
    }
  }, [recordAudio]);

  const askQuestion = useCallback(async () => {
    try {
      await recordAudio("question");
    } catch (error) {
      // Error already handled in recordAudio function
      console.error("Unable to start recording question:", error);
    }
  }, [recordAudio]);

  const stopRecording = useCallback(async () => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }

    setIsRecording(false);
    setRecordingTime(0);

    // Clean up audio resources
    if (recorderNodeRef.current) {
      try {
        recorderNodeRef.current.disconnect();
        recorderNodeRef.current.port.postMessage({ command: "stop" });
        recorderNodeRef.current = null;
      } catch (error) {
        console.error("Error stopping recorder node:", error);
      }
    }

    if (audioCtxRef.current) {
      audioCtxRef.current.close().catch(console.error);
      audioCtxRef.current = null;
    }

    // Notify server to end session
    if (sessionId.current) {
      try {
        await fetch(`/api/transcribe?sessionId=${sessionId.current}`, {
          method: "DELETE",
        });
      } catch (error) {
        console.error("Error ending session:", error);
      }
    }

    // Handle different modes
    if (mode === "question") {
      setWaitingForAnswer(true);
      waitingForAnswerRef.current = true;
      hasReceivedAnswerRef.current = false;
      toast.info("Processing your question...");
    } else {
      setMode(undefined);
    }
  }, [mode, setMode, setIsRecording, setRecordingTime]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (recorderNodeRef.current) {
        try {
          recorderNodeRef.current.disconnect();
          recorderNodeRef.current.port.postMessage({ command: "stop" });
        } catch (error) {
          console.error("Error cleaning up recorder node:", error);
        }
      }
      if (audioCtxRef.current) {
        audioCtxRef.current.close().catch(console.error);
      }
      if (eventSource.current) {
        eventSource.current.close();
      }
      // Clear any existing timeouts
      if (connectionTimeoutRef.current !== null) {
        clearTimeout(connectionTimeoutRef.current);
        connectionTimeoutRef.current = null;
      }
    };
  }, []);

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
      {!isRecording && !waitingForAnswer ? (
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
      ) : isRecording ? (
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
      ) : waitingForAnswer ? (
        <div className="p-4 bg-card border border-border/50 rounded-full flex items-center gap-4 max-w-4xl">
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-muted-foreground">
                Processing your question...
              </span>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
