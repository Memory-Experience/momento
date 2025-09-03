"use client";
import { useCallback, useEffect, useRef } from "react";
import TranscriptionService from "@/services/TranscriptionService";
import { MemoryChunk } from "protos/generated/ts/stt";
import { Mic } from "lucide-react";
import AudioRecorder from "./AudioRecorder";
import TranscriptionMessage from "../messages/TranscriptionMessage";
import { Key, ReactNode } from "react";

interface MemoryRecorderProps {
  onMemoryStart: () => void;
  onMemoryEnd: () => void;
  onAddMessage: (message: ReactNode) => void;
  onSessionError: (errorMessage?: string) => void;
  transcriptionService: TranscriptionService;
  isRecording: boolean;
  isDisabled?: boolean;
}

export default function MemoryRecorder({
  onMemoryStart,
  onMemoryEnd,
  onAddMessage,
  onSessionError,
  transcriptionService,
  isRecording,
  isDisabled = false,
}: MemoryRecorderProps) {
  const messageIdCounterRef = useRef(0);
  const audioChunkCountRef = useRef(0);

  // Create a message component from a chunk
  const createMessageFromChunk = useCallback(
    (chunk: MemoryChunk): ReactNode => {
      const id = `memory_${messageIdCounterRef.current++}`;
      return <TranscriptionMessage key={id as Key} chunk={chunk} />;
    },
    [],
  );

  // Process incoming chunks
  const handleChunk = useCallback(
    (chunk: MemoryChunk) => {
      // Skip chunks with no text data
      console.log("Received MemoryChunk", chunk.textData);

      if (!chunk.textData) return;

      // Create and add the message component
      const messageComponent = createMessageFromChunk(chunk);
      onAddMessage(messageComponent);
    },
    [createMessageFromChunk, onAddMessage],
  );

  // Handle session status
  const handleSessionStatus = useCallback(
    (connected: boolean, errorMessage?: string) => {
      if (connected) {
        console.log("MemoryRecorder: Session connected successfully");
      } else if (errorMessage) {
        console.error(`MemoryRecorder: Session error: ${errorMessage}`);
        onSessionError(errorMessage);
        onMemoryEnd();
      }
    },
    [onMemoryEnd, onSessionError],
  );

  // Handle audio data
  const handleAudioData = useCallback(
    async (audioData: Uint8Array) => {
      audioChunkCountRef.current++;

      if (audioChunkCountRef.current % 10 === 0) {
        console.log(
          `MemoryRecorder: Recording=${isRecording}, Received ${audioChunkCountRef.current} audio chunks`,
        );
      }

      if (isRecording) {
        const success = await transcriptionService.sendAudioData(audioData);

        if (!success && audioChunkCountRef.current % 10 === 0) {
          console.warn("MemoryRecorder: Failed to send audio data to server");
        }
      }
    },
    [transcriptionService, isRecording],
  );

  // Start recording handler
  const handleStartRecording = useCallback(() => {
    console.log("MemoryRecorder: Start recording triggered");

    // Signal start of recording immediately to update UI state
    onMemoryStart();

    // Reset counter for this session
    messageIdCounterRef.current = 0;
    audioChunkCountRef.current = 0;

    // Start the memory session
    const success = transcriptionService.startRecordingSession(
      "memory",
      handleChunk,
      handleSessionStatus,
    );

    if (!success) {
      console.error("MemoryRecorder: Failed to start session");
      // If session failed to start, end the memory recording
      onMemoryEnd();
    }
  }, [
    transcriptionService,
    handleChunk,
    handleSessionStatus,
    onMemoryStart,
    onMemoryEnd,
  ]);

  // Cleanup on unmount if still recording
  useEffect(() => {
    return () => {
      if (isRecording) {
        transcriptionService.endSession();
      }
    };
  }, [isRecording, transcriptionService]);

  // Debug logging for prop changes
  useEffect(() => {
    console.log(`MemoryRecorder: isRecording changed to ${isRecording}`);
  }, [isRecording]);

  return (
    <AudioRecorder
      buttonText="Record a Memory"
      buttonIcon={<Mic className="mr-2 h-4 w-4" />}
      onAudioData={handleAudioData}
      onStartRecording={handleStartRecording}
      isRecordingActive={isRecording}
      disabled={isDisabled}
    />
  );
}
