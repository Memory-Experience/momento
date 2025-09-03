"use client";
import { useCallback, useEffect, useRef } from "react";
import TranscriptionService from "@/services/TranscriptionService";
import { ChunkType, MemoryChunk } from "protos/generated/ts/stt";
import { MessageCircleQuestionMark } from "lucide-react";
import AudioRecorder from "./AudioRecorder";
import TranscriptionMessage from "../messages/TranscriptionMessage";
import AnswerMessage from "../messages/AnswerMessage";
import { Key, ReactNode } from "react";

interface QuestionRecorderProps {
  onQuestionStart: () => void;
  onQuestionEnd: () => void;
  onProcessingStart: () => void;
  onAddMessage: (message: ReactNode) => void;
  onAnswerReceived: () => void;
  onSessionError: (errorMessage?: string) => void;
  transcriptionService: TranscriptionService;
  isRecording: boolean;
  isDisabled?: boolean;
}

export default function QuestionRecorder({
  onQuestionStart,
  onQuestionEnd,
  onAddMessage,
  onAnswerReceived,
  onSessionError,
  transcriptionService,
  isRecording,
  isDisabled = false,
}: QuestionRecorderProps) {
  const messageIdCounterRef = useRef(0);
  const audioChunkCountRef = useRef(0);

  // Create a message component from a chunk
  const createMessageFromChunk = useCallback(
    (chunk: MemoryChunk): ReactNode | null => {
      if (!chunk.metadata?.type) return null;

      const id = `question_${messageIdCounterRef.current++}`;

      switch (chunk.metadata.type) {
        case ChunkType.TRANSCRIPT:
          return <TranscriptionMessage key={id as Key} chunk={chunk} />;
        case ChunkType.ANSWER:
          return <AnswerMessage key={id as Key} chunk={chunk} />;
        default:
          return null;
      }
    },
    [],
  );

  // Process incoming chunks
  const handleChunk = useCallback(
    (chunk: MemoryChunk) => {
      // Skip chunks with no text data
      if (!chunk.textData) return;

      // Create and add the message component
      const messageComponent = createMessageFromChunk(chunk);
      if (messageComponent) {
        onAddMessage(messageComponent);

        // Check if this is an answer message to end processing
        if (chunk.metadata?.type === ChunkType.ANSWER) {
          onAnswerReceived();
        }
      }
    },
    [createMessageFromChunk, onAddMessage, onAnswerReceived],
  );

  // Handle session status
  const handleSessionStatus = useCallback(
    (connected: boolean, errorMessage?: string) => {
      if (connected) {
        console.log("QuestionRecorder: Session connected successfully");
      } else if (errorMessage) {
        console.error(`QuestionRecorder: Session error: ${errorMessage}`);
        onSessionError(errorMessage);
        onQuestionEnd();
      }
    },
    [onQuestionEnd, onSessionError],
  );

  // Handle audio data
  const handleAudioData = useCallback(
    async (audioData: Uint8Array) => {
      audioChunkCountRef.current++;

      if (audioChunkCountRef.current % 10 === 0) {
        console.log(
          `QuestionRecorder: Received ${audioChunkCountRef.current} audio chunks`,
        );
      }

      if (isRecording) {
        const success = await transcriptionService.sendAudioData(audioData);

        if (!success && audioChunkCountRef.current % 10 === 0) {
          console.warn("QuestionRecorder: Failed to send audio data to server");
        }
      }
    },
    [transcriptionService, isRecording],
  );

  // Start recording handler
  const handleStartRecording = useCallback(() => {
    // Reset counter for this session
    messageIdCounterRef.current = 0;
    audioChunkCountRef.current = 0;

    // Start the question session
    const success = transcriptionService.startRecordingSession(
      "question",
      handleChunk,
      handleSessionStatus,
    );

    if (success) {
      onQuestionStart();
    }
  }, [transcriptionService, handleChunk, handleSessionStatus, onQuestionStart]);

  // Cleanup on unmount if still recording
  useEffect(() => {
    return () => {
      if (isRecording) {
        transcriptionService.endSession();
      }
    };
  }, [isRecording, transcriptionService]);

  return (
    <AudioRecorder
      buttonText="Ask a Question"
      buttonIcon={<MessageCircleQuestionMark className="mr-2 h-4 w-4" />}
      onAudioData={handleAudioData}
      onStartRecording={handleStartRecording}
      isRecordingActive={isRecording}
      disabled={isDisabled}
    />
  );
}
