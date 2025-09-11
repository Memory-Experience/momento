"use client";

import { useState, useRef, useEffect, ReactNode, useCallback } from "react";
import { ChatContext } from "@/context/ChatContext";
import Messages from "./Messages";
import MemoryRecorder from "./controls/MemoryRecorder";
import QuestionRecorder from "./controls/QuestionRecorder";
import { cn } from "@/utils";
import { toast } from "sonner";
import BackendService from "@/services/BackendService";
import RecordingIndicator from "./RecordingIndicator";
import ConnectionIndicator from "./ConnectionIndicator";

export default function Chat() {
  const [mode, setMode] = useState<"memory" | "question" | undefined>(
    undefined,
  );
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [messages, setMessages] = useState<ReactNode[]>([]);
  const [transcriptionService] = useState(() => new BackendService());
  const [isConnected, setIsConnected] = useState(false);
  const [isCheckingConnection, setIsCheckingConnection] = useState(true);

  const messagesContainerRef = useRef<HTMLDivElement>(null);

  // Set up connection management
  useEffect(() => {
    // Register error handler
    transcriptionService.registerErrorHandler((message) => {
      toast.error(message);
    });

    // Register connection status handler
    transcriptionService.registerConnectionStatusHandler((connected) => {
      setIsConnected(connected);
    });

    // Initialize the service and check connection
    const checkConnection = async () => {
      setIsCheckingConnection(true);
      try {
        const isAvailable = await transcriptionService.initialize();
        setIsConnected(isAvailable);
      } catch (error) {
        console.warn(error);
        setIsConnected(false);
      } finally {
        setIsCheckingConnection(false);
      }
    };

    checkConnection();

    return () => {
      // Cleanup on unmount
      transcriptionService.reset();
    };
  }, [transcriptionService]);

  // Handle memory mode events
  const handleMemoryStart = useCallback(() => {
    console.log("Chat: Memory start triggered");
    setMessages([]);
    setMode("memory");
    setIsRecording(true);
  }, []);

  const handleMemoryEnd = useCallback(() => {
    console.log("Chat: Memory end triggered");
    setIsRecording(false);
    setMode(undefined);
  }, []);

  // Handle question mode events
  const handleQuestionStart = useCallback(() => {
    console.log("Chat: Question start triggered");
    setMessages([]);
    setMode("question");
    setIsRecording(true);
  }, []);

  const handleQuestionEnd = useCallback(() => {
    console.log("Chat: Question end triggered");
    setIsRecording(false);
    setIsProcessing(false);
  }, []);

  const handleProcessingStart = () => {
    setIsProcessing(true);
  };

  // Handle answer received
  const handleAnswerReceived = () => {
    setIsProcessing(false);
    setMode(undefined);
  };

  // Add a message to the messages list
  const handleAddMessage = (message: ReactNode) => {
    setMessages((prev) => [...prev, message]);
  };

  // Handle session error
  const handleSessionError = (errorMessage?: string) => {
    if (errorMessage) {
      toast.error(errorMessage);
    }
  };

  useEffect(() => {
    const handleBeforeUnload = () => {
      console.log("Page is about to unload, cleaning up transcription service");

      if (transcriptionService.hasActiveSession()) {
        // This will ensure EventSource is properly closed before the page refreshes
        transcriptionService.closeEventSourceConnection();
      }
    };

    // Add the event listener
    window.addEventListener("beforeunload", handleBeforeUnload);

    // Clean up on component unmount
    return () => {
      window.removeEventListener("beforeunload", handleBeforeUnload);
    };
  }, [transcriptionService]);

  return (
    <ChatContext.Provider
      value={{
        mode,
        setMode,
        isRecording,
        setIsRecording,
        isProcessing,
        setIsProcessing,
        messages,
        setMessages,
      }}
    >
      <div className="flex flex-col h-full">
        <div className="fixed bottom-4 right-4 z-40">
          <ConnectionIndicator
            isConnected={isConnected}
            isChecking={isCheckingConnection}
          />
        </div>

        <Messages ref={messagesContainerRef} messages={messages} mode={mode} />

        {/* Background gradient that extends to the bottom */}
        <div className="fixed bottom-0 left-0 right-0 w-full h-64 bg-gradient-to-t from-card via-card/90 to-transparent z-10"></div>

        {/* Fixed position control area */}
        <div className="fixed bottom-32 left-0 right-0 w-full p-4 pb-8 flex flex-col items-center justify-center gap-4 z-20">
          {/* Recording indicator */}
          {isRecording && (
            <RecordingIndicator
              isRecording={isRecording}
              onStopRecording={() => {
                if (mode === "memory") {
                  handleMemoryEnd();
                } else if (mode === "question") {
                  handleQuestionEnd();
                }
              }}
            />
          )}

          {/* Processing indicator */}
          {!isRecording && isProcessing && (
            <div
              className={cn(
                "p-4 bg-card border border-border/50 rounded-full",
                "flex items-center gap-2 max-w-md mx-auto",
              )}
            >
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              <span className="text-sm text-muted-foreground">
                Processing your question...
              </span>
            </div>
          )}

          {/* Backend unavailable warning */}
          {!isConnected &&
            !isRecording &&
            !isProcessing &&
            !isCheckingConnection && (
              <div
                className={cn(
                  "p-4 bg-red-100 border border-red-300 rounded-lg text-red-800",
                  "flex items-center gap-2 max-w-md mx-auto",
                )}
              >
                <span>
                  Backend service unavailable. Please try again later.
                </span>
              </div>
            )}

          {/* Recording buttons container */}
          <div
            className={cn(
              "max-w-md mx-auto",
              "flex flex-row items-center justify-center gap-4",
              "z-30",
            )}
          >
            <div className="flex-1">
              <MemoryRecorder
                onMemoryStart={handleMemoryStart}
                onMemoryEnd={handleMemoryEnd}
                onAddMessage={handleAddMessage}
                onSessionError={handleSessionError}
                transcriptionService={transcriptionService}
                isRecording={isRecording && mode === "memory"}
                isDisabled={isRecording || isProcessing}
              />
            </div>

            <div className="flex-1">
              <QuestionRecorder
                onQuestionStart={handleQuestionStart}
                onQuestionEnd={handleQuestionEnd}
                onProcessingStart={handleProcessingStart}
                onAddMessage={handleAddMessage}
                onAnswerReceived={handleAnswerReceived}
                onSessionError={handleSessionError}
                transcriptionService={transcriptionService}
                isRecording={isRecording && mode === "question"}
                isDisabled={isRecording || isProcessing}
              />
            </div>
          </div>
        </div>
      </div>
    </ChatContext.Provider>
  );
}
