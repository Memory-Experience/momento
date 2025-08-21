import { createContext, Dispatch, SetStateAction } from "react";

export const ChatContext = createContext<{
  mode: "memory" | "question" | undefined;
  setMode: Dispatch<SetStateAction<"memory" | "question" | undefined>>;
  isRecording: boolean;
  setIsRecording: Dispatch<SetStateAction<boolean>>;
  transcriptions: string[];
  setTranscriptions: Dispatch<SetStateAction<string[]>>;
}>({
  mode: undefined,
  setMode: () => {},
  isRecording: false,
  setIsRecording: () => {},
  transcriptions: [],
  setTranscriptions: () => {},
});
