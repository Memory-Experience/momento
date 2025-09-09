"use client";
import React, { useCallback, useEffect, useRef, useState } from "react";
import TranscriptionService from "@/services/TranscriptionService";
import { ChunkType, MemoryChunk } from "protos/generated/ts/stt";
import { MessageCircleQuestionMark } from "lucide-react";
import AudioRecorder from "./AudioRecorder";
import StreamingMessage, {
  StreamingMessageHandle,
} from "../messages/StreamingMessage";
import AnswerMessage, { AnswerMessageHandle } from "../messages/AnswerMessage";
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
  onProcessingStart,
  onAddMessage,
  onAnswerReceived,
  onSessionError,
  transcriptionService,
  isRecording,
  isDisabled = false,
}: QuestionRecorderProps) {
  const audioChunkCountRef = useRef(0);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [isWaitingForAnswer, setIsWaitingForAnswer] = useState(false);
  const [transcriptionComplete, setTranscriptionComplete] =
    useState<boolean>(false);

  // Store references to message components
  const transcriptionRef = useRef<StreamingMessageHandle | null>(null);
  const answerRef = useRef<AnswerMessageHandle | null>(null);
  const answerTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const isAnswerComponentCreatedRef = useRef(false);
  const earlyMemoryChunksRef = useRef<MemoryChunk[]>([]); // Buffer for early memory chunks

  // Handle transcription completion
  const handleTranscriptionComplete = useCallback(() => {
    console.log("QuestionRecorder: Transcription marked as complete");
    setTranscriptionComplete(true);

    // If we're not recording, start waiting for the answer
    if (!isRecording && isSessionActive && !isWaitingForAnswer) {
      console.log(
        "Recording stopped and transcription complete, waiting for answer...",
      );
      setIsWaitingForAnswer(true);
      onProcessingStart();
    }
  }, [isRecording, isSessionActive, isWaitingForAnswer, onProcessingStart]);

  // Function to finalize answer and clean up
  const finalizeAnswer = useCallback(() => {
    if (answerTimeoutRef.current) {
      clearTimeout(answerTimeoutRef.current);
      answerTimeoutRef.current = null;
    }

    setIsWaitingForAnswer(false);

    // End the session and clean up
    if (isSessionActive) {
      setIsSessionActive(false);
      transcriptionService.endSession();
      onQuestionEnd();
    }

    // Reset ALL state for next question
    answerRef.current = null;
    earlyMemoryChunksRef.current = []; // Clear any buffered memory chunks
    isAnswerComponentCreatedRef.current = false; // Reset component creation flag
  }, [transcriptionService, onQuestionEnd, isSessionActive]);

  // Handle answer completion
  const handleAnswerComplete = useCallback(() => {
    console.log("QuestionRecorder: Answer marked as complete");
    finalizeAnswer();
  }, [finalizeAnswer]);

  // Process incoming chunks
  const handleChunk = useCallback(
    (chunk: MemoryChunk) => {
      if (chunk.metadata?.type === ChunkType.TRANSCRIPT) {
        // Update the transcription message with new transcript chunks
        if (transcriptionRef.current) {
          transcriptionRef.current.processChunk(chunk);
        }
        return;
      }

      if (
        chunk.metadata?.type === ChunkType.ANSWER ||
        chunk.metadata?.type === ChunkType.MEMORY
      ) {
        // Create answer component on first content chunk if it doesn't exist yet
        if (!isAnswerComponentCreatedRef.current) {
          isAnswerComponentCreatedRef.current = true;
          const answerComponent = (
            <AnswerMessage
              key={`answer_${Date.now()}` as Key}
              ref={(instance) => {
                answerRef.current = instance;
                // Process any buffered memory chunks when the ref is available
                if (instance && earlyMemoryChunksRef.current.length > 0) {
                  console.log(
                    `Processing ${earlyMemoryChunksRef.current.length} buffered memory chunks`,
                  );
                  earlyMemoryChunksRef.current.forEach((bufferedChunk) => {
                    instance.processChunk(bufferedChunk);
                  });
                  earlyMemoryChunksRef.current = [];
                }
              }}
              onComplete={handleAnswerComplete}
            />
          );

          onAddMessage(answerComponent);
          onAnswerReceived();
        }

        // Process the chunk in the answer component if it exists
        if (answerRef.current) {
          answerRef.current.processChunk(chunk);
        } else {
          // Buffer the chunk for later processing
          earlyMemoryChunksRef.current.push(chunk);
          console.log(
            `Buffering early ${chunk.metadata?.type} chunk for later processing`,
          );
        }

        // Clear any inactivity timer since we got data
        if (answerTimeoutRef.current) {
          clearTimeout(answerTimeoutRef.current);
          answerTimeoutRef.current = null;
        }
      }
    },
    [onAddMessage, onAnswerReceived, handleAnswerComplete],
  );

  // Handle session status
  const handleSessionStatus = useCallback(
    (connected: boolean, errorMessage?: string) => {
      if (connected) {
        console.log("QuestionRecorder: Session connected successfully");
        setIsSessionActive(true);
      } else {
        // Connection ended but may not be an error
        if (isWaitingForAnswer && !errorMessage) {
          console.log(
            "QuestionRecorder: Server closed connection, checking answer status",
          );

          // If we have an answer component but it's not marked as complete,
          // we should mark it complete now
          if (answerRef.current) {
            console.log(
              "QuestionRecorder: Marking answer as complete due to server disconnect",
            );
            answerRef.current.markComplete();
          } else {
            console.log(
              "QuestionRecorder: No answer component found, finalizing directly",
            );
            finalizeAnswer();
          }
        }
        // Handle actual errors
        else if (errorMessage) {
          console.error(`QuestionRecorder: Session error: ${errorMessage}`);
          onSessionError(errorMessage);
          setIsSessionActive(false);
          setIsWaitingForAnswer(false);
          onQuestionEnd();
        }
        // Always reset session state when disconnected, even if it wasn't an answer session
        else {
          console.log("QuestionRecorder: Session ended, resetting state");
          setIsSessionActive(false);
          setTranscriptionComplete(false);
          setIsWaitingForAnswer(false);
        }
      }
    },
    [onQuestionEnd, onSessionError, isWaitingForAnswer, finalizeAnswer],
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

      if (isRecording && isSessionActive) {
        const success = await transcriptionService.sendAudioData(audioData);

        if (!success && audioChunkCountRef.current % 10 === 0) {
          console.warn("QuestionRecorder: Failed to send audio data to server");
        }
      }
    },
    [transcriptionService, isRecording, isSessionActive],
  );

  // Start recording handler
  const handleStartRecording = useCallback(async () => {
    // First, ensure any previous session is completely terminated
    if (isSessionActive) {
      console.log(
        "QuestionRecorder: Terminating previous session before starting new one",
      );
      transcriptionService.endSession();
      setIsSessionActive(false);

      // Wait a brief moment for session termination to complete
      await new Promise((resolve) => setTimeout(resolve, 100));
    }

    // Reset ALL state for this session
    console.log("QuestionRecorder: Resetting all state for new session");
    audioChunkCountRef.current = 0;
    setTranscriptionComplete(false);
    setIsWaitingForAnswer(false);
    isAnswerComponentCreatedRef.current = false;
    earlyMemoryChunksRef.current = [];
    transcriptionRef.current = null;
    answerRef.current = null;

    // Start the question session
    const success = await transcriptionService.startRecordingSession(
      "question",
      handleChunk,
      handleSessionStatus,
    );

    if (success) {
      onQuestionStart();

      // Create and add the transcription component with a stable key
      const key = `transcript_${Date.now()}`;
      const transcriptionComponent = (
        <StreamingMessage
          key={key}
          ref={(instance) => {
            transcriptionRef.current = instance;
          }}
          initialContent=""
          onComplete={handleTranscriptionComplete}
        />
      );

      onAddMessage(transcriptionComponent);
    }
  }, [
    transcriptionService,
    handleChunk,
    handleSessionStatus,
    onQuestionStart,
    onAddMessage,
    handleTranscriptionComplete,
    isSessionActive, // Add this dependency
  ]);

  // Effect to handle when recording stops
  useEffect(() => {
    // When recording stops, send a final marker to the backend
    if (!isRecording && isSessionActive && !transcriptionComplete) {
      console.log(
        "Question recording stopped, sending final transcription marker",
      );
      transcriptionService.sendFinalMarker().then((success) => {
        if (!success) {
          console.error("Failed to send final marker for question");
          onSessionError("Failed to send final marker for question");
        }
      });
    }
  }, [
    isRecording,
    isSessionActive,
    transcriptionComplete,
    transcriptionService,
    onSessionError,
  ]);

  // Add a safety timeout for waiting for answers (optional)
  useEffect(() => {
    let timeoutId: NodeJS.Timeout | undefined;

    if (isWaitingForAnswer) {
      // Set a timeout to end the session if no answer is received after a while (e.g., 30 seconds)
      timeoutId = setTimeout(() => {
        if (isWaitingForAnswer && isSessionActive) {
          console.warn(
            "No answer received within timeout period, ending session",
          );
          setIsWaitingForAnswer(false);
          setIsSessionActive(false);
          transcriptionService.endSession();
          onSessionError("No answer received within the expected time");
          onQuestionEnd();
        }
      }, 30000); // 30 seconds timeout
    }

    return () => {
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [
    isWaitingForAnswer,
    isSessionActive,
    transcriptionService,
    onSessionError,
    onQuestionEnd,
  ]);

  // Cleanup on unmount if still active
  useEffect(() => {
    return () => {
      if (isSessionActive) {
        transcriptionService.endSession();
      }
    };
  }, [isSessionActive, transcriptionService]);

  // Add a specific handler to fully reset state when question ends
  useEffect(() => {
    if (!isSessionActive && transcriptionComplete) {
      // Reset state after session ends and transcription was completed
      console.log(
        "QuestionRecorder: Full state reset after session completion",
      );
      setTranscriptionComplete(false);
      setIsWaitingForAnswer(false);
      earlyMemoryChunksRef.current = [];
      isAnswerComponentCreatedRef.current = false;
    }
  }, [isSessionActive, transcriptionComplete]);

  return (
    <AudioRecorder
      buttonText="Ask a Question"
      buttonIcon={<MessageCircleQuestionMark className="mr-2 h-4 w-4" />}
      onAudioData={handleAudioData}
      onStartRecording={handleStartRecording}
      isRecordingActive={isRecording}
      disabled={isDisabled || isWaitingForAnswer}
    />
  );
}
