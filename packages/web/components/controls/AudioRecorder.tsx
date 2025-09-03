"use client";

import { cn } from "@/utils";
import { Mic } from "lucide-react";
import { Button } from "../ui/button";
import { useCallback, useState, useEffect, useRef } from "react";
import { toast } from "sonner";

interface AudioRecorderProps {
  buttonText: string;
  buttonIcon?: React.ReactNode;
  onAudioData: (audioData: Uint8Array) => Promise<void>;
  onStartRecording?: () => void;
  className?: string;
  isRecordingActive?: boolean;
  disabled?: boolean;
}

export default function AudioRecorder({
  buttonText,
  buttonIcon = <Mic className="mr-2 h-4 w-4" />,
  onAudioData,
  onStartRecording,
  className,
  isRecordingActive = false,
  disabled = false,
}: AudioRecorderProps) {
  const [isInternalRecording, setIsInternalRecording] = useState(false);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const recorderNodeRef = useRef<AudioWorkletNode | null>(null);
  const audioChunkCountRef = useRef(0);
  const audioDataCallbackRef = useRef(onAudioData);

  // Update the ref when the prop changes
  useEffect(() => {
    audioDataCallbackRef.current = onAudioData;
  }, [onAudioData]);

  // Update the handleAudioData to use the ref
  const handleAudioData = useCallback(async (pcmBytes: Uint8Array) => {
    if (pcmBytes.byteLength < 10) return;

    audioChunkCountRef.current++;
    if (audioChunkCountRef.current % 10 === 0) {
      console.log(
        `AudioRecorder: Received ${audioChunkCountRef.current} audio chunks`,
      );
    }

    // Always use the latest callback via the ref
    await audioDataCallbackRef.current(pcmBytes);
  }, []);

  // Start recording
  const startAudioRecording = useCallback(async (): Promise<boolean> => {
    try {
      // Reset audio chunk counter
      audioChunkCountRef.current = 0;
      console.log("AudioRecorder: Starting recording...");

      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true },
      });

      // Create and configure AudioContext
      const audioCtx = new AudioContext({ latencyHint: "interactive" });
      audioCtxRef.current = audioCtx;

      if (audioCtx.state === "suspended") await audioCtx.resume();
      console.log(
        `AudioRecorder: Audio context created, state: ${audioCtx.state}, sample rate: ${audioCtx.sampleRate}`,
      );

      // Create audio source from microphone stream
      const source = audioCtx.createMediaStreamSource(stream);

      // Add audio worklet for processing
      await audioCtx.audioWorklet.addModule(
        "/worklets/audio-recorder.worklet.js",
      );
      console.log("AudioRecorder: Audio worklet module loaded");

      // Create recorder node with specified options
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

      // Handle messages from audio worklet processor
      recorderNode.port.onmessage = (event: MessageEvent) => {
        if (event.data?.type) {
          console.log(
            `AudioRecorder: Received message type: ${event.data.type}`,
          );
          return; // Skip debug/level messages
        }
        const pcmBytes = event.data;
        if (pcmBytes instanceof Uint8Array && pcmBytes.byteLength > 0) {
          handleAudioData(new Uint8Array(pcmBytes));
        } else {
          console.warn(
            "AudioRecorder: Received invalid audio data",
            typeof pcmBytes,
            pcmBytes,
          );
        }
      };

      // Initialize audio worklet
      recorderNode.port.postMessage({
        command: "init",
        config: {
          bufferDuration: 100, // 100ms buffer - should send audio chunks every 100ms
          nativeSampleRate: audioCtx.sampleRate,
          targetSampleRate: 16000,
        },
      });
      console.log("AudioRecorder: Worklet initialized with 100ms buffer");

      // Connect audio nodes
      source.connect(recorderNode);
      const silentGain = audioCtx.createGain();
      silentGain.gain.value = 0;
      recorderNode.connect(silentGain).connect(audioCtx.destination);
      console.log("AudioRecorder: Audio nodes connected");

      return true;
    } catch (error) {
      console.error("Error starting audio recording:", error);
      return false;
    }
  }, [handleAudioData]);

  // Stop recording
  const stopAudioRecording = useCallback(() => {
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
  }, []);

  // Start recording
  const startRecording = useCallback(async () => {
    // Start audio recording
    const success = await startAudioRecording();
    if (success) {
      setIsInternalRecording(true);
      if (onStartRecording) {
        onStartRecording();
      }
    } else {
      toast.error(
        "Failed to start recording. Please check microphone permissions.",
      );
    }
  }, [startAudioRecording, onStartRecording]);

  // Sync with external isRecordingActive state
  useEffect(() => {
    if (!isInternalRecording && isRecordingActive) {
      // Parent wants to start recording
      startAudioRecording();
      setIsInternalRecording(true);
    } else if (isInternalRecording && !isRecordingActive) {
      // Parent wants to stop recording
      stopAudioRecording();
      setIsInternalRecording(false);
    }
  }, [
    isRecordingActive,
    isInternalRecording,
    startAudioRecording,
    stopAudioRecording,
  ]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
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
    };
  }, []);

  return (
    <div className={cn(className)}>
      <Button
        onClick={startRecording}
        className={cn(
          "rounded-full",
          isRecordingActive && "bg-red-300 hover:bg-red-600 animate-pulse",
        )}
        size="lg"
        disabled={disabled || isRecordingActive || isInternalRecording}
      >
        {buttonIcon}
        {buttonText}
      </Button>
    </div>
  );
}
