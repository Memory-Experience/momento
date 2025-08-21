"use client";

import AudioRecorder from "@/components/AudioRecorder";
import Messages from "./Messages";
import { useState } from "react";
import { ChatContext } from "@/context/ChatContext";

export default function Chat() {
  const [mode, setMode] = useState<"memory" | "question" | undefined>(
    undefined,
  );
  const [isRecording, setIsRecording] = useState(false);
  const [transcriptions, setTranscriptions] = useState<string[]>([]);

  return (
    <ChatContext.Provider
      value={{
        mode,
        setMode,
        isRecording,
        setIsRecording,
        transcriptions,
        setTranscriptions,
      }}
    >
      <Messages transcriptions={transcriptions} />
      <AudioRecorder />
    </ChatContext.Provider>
  );
}
