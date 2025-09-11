"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import BackendService from "@/services/BackendService";
import { MemoryChunk, ChunkType } from "protos/generated/ts/stt";
import { Mic } from "lucide-react";
import AudioRecorder from "./AudioRecorder";
import StreamingMessage, {
  StreamingMessageHandle,
} from "../messages/StreamingMessage";
import { ReactNode } from "react";
import { toast } from "sonner";

interface MemoryRecorderProps {
  onMemoryStart: () => void;
  onMemoryEnd: () => void;
  onAddMessage: (message: ReactNode) => void;
  onSessionError: (errorMessage?: string) => void;
  transcriptionService: BackendService;
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
  const audioChunkCountRef = useRef(0);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [transcriptionComplete, setTranscriptionComplete] =
    useState<boolean>(false);
  const [memorySaved, setMemorySaved] = useState<boolean>(false);
  const [memoryId, setMemoryId] = useState<string | null>(null);

  // Store reference to the transcription message component
  const streamingMessageRef = useRef<StreamingMessageHandle | null>(null);

  // Handle transcription completion
  const handleTranscriptionComplete = useCallback(() => {
    console.log("MemoryRecorder: Transcription marked as complete");
    setTranscriptionComplete(true);
  }, []);

  // Handle memory saved event
  const handleMemorySaved = useCallback(
    (id: string) => {
      console.log(`MemoryRecorder: Memory saved with ID: ${id}`);
      setMemoryId(id);
      console.log(memoryId);
      setMemorySaved(true);

      // Now we can end the session
      if (isSessionActive) {
        console.log("MemoryRecorder: Memory saved, ending session");
        transcriptionService.endSession();
      }

      setIsSessionActive(false);
      onMemoryEnd();

      toast.success("Memory successfully recorded and saved", {
        description:
          "Your memory has been transcribed and stored in the system.",
      });
    },
    [isSessionActive, transcriptionService, onMemoryEnd, memoryId],
  );

  // Process incoming chunks
  const handleChunk = useCallback(
    (chunk: MemoryChunk) => {
      // Skip chunks with no text data unless it's a final marker
      if (
        !chunk.textData &&
        !chunk.metadata?.isFinal &&
        chunk.metadata?.type !== ChunkType.MEMORY
      )
        return;

      // Check if this is a Memory chunk containing saved memory ID
      if (chunk.metadata?.type === ChunkType.MEMORY) {
        console.log(
          "Received MEMORY chunk with saved memory confirmation:",
          chunk.metadata.memoryId,
        );
        handleMemorySaved(chunk.metadata.memoryId);
        return;
      }

      // Update the existing streaming message
      if (streamingMessageRef.current) {
        streamingMessageRef.current.processChunk(chunk);
      } else {
        console.warn("StreamingMessage ref is null, can't process chunk");
      }
    },
    [handleMemorySaved],
  );

  // Handle session status
  const handleSessionStatus = useCallback(
    (connected: boolean, errorMessage?: string) => {
      if (connected) {
        console.log("MemoryRecorder: Session connected successfully");
        setIsSessionActive(true);
      } else if (errorMessage) {
        console.error(`MemoryRecorder: Session error: ${errorMessage}`);
        onSessionError(errorMessage);
        setIsSessionActive(false);
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

      if (isRecording && isSessionActive) {
        const success = await transcriptionService.sendAudioData(audioData);

        if (!success && audioChunkCountRef.current % 10 === 0) {
          console.warn("MemoryRecorder: Failed to send audio data to server");
        }
      }
    },
    [transcriptionService, isRecording, isSessionActive],
  );

  // Handle recording stop - send final flag but don't close session yet
  useEffect(() => {
    if (!isRecording && isSessionActive && !transcriptionComplete) {
      console.log(
        "Memory recording stopped, sending final transcription marker",
      );

      // Send a message to mark the end of the audio stream
      // This is done via a custom function in the transcription service
      transcriptionService.sendFinalMarker().then((success) => {
        if (!success) {
          console.error("Failed to send final marker");
        }
      });
    }
  }, [
    isRecording,
    isSessionActive,
    transcriptionComplete,
    transcriptionService,
  ]);

  // Start recording handler
  const handleStartRecording = useCallback(async () => {
    console.log("MemoryRecorder: Start recording triggered");

    // Reset state and refs
    audioChunkCountRef.current = 0;
    setTranscriptionComplete(false);
    setMemorySaved(false);
    setMemoryId(null);
    streamingMessageRef.current = null;

    // Start the memory session
    const success = await transcriptionService.startRecordingSession(
      "memory",
      handleChunk,
      handleSessionStatus,
    );

    if (success) {
      // Signal start of recording to update UI state
      onMemoryStart();

      // Create and add the streaming message component with a stable key
      const key = `memory_${Date.now()}`;
      const streamingComponent = (
        <StreamingMessage
          key={key}
          ref={(instance) => {
            streamingMessageRef.current = instance;
          }}
          initialContent=""
          onComplete={handleTranscriptionComplete}
        />
      );

      onAddMessage(streamingComponent);
    } else {
      console.error("MemoryRecorder: Failed to start session");
      onMemoryEnd();
    }
  }, [
    transcriptionService,
    handleChunk,
    handleSessionStatus,
    onMemoryStart,
    onMemoryEnd,
    onAddMessage,
    handleTranscriptionComplete,
  ]);

  // Cleanup on unmount if still active
  useEffect(() => {
    return () => {
      if (isSessionActive) {
        transcriptionService.endSession();
      }
    };
  }, [isSessionActive, transcriptionService]);

  return (
    <AudioRecorder
      buttonText="Record a Memory"
      buttonIcon={<Mic className="mr-2 h-4 w-4" />}
      onAudioData={handleAudioData}
      onStartRecording={handleStartRecording}
      isRecordingActive={isRecording}
      disabled={isDisabled || (!isRecording && isSessionActive && !memorySaved)}
    />
  );
}
