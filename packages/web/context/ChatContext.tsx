import { createContext, Dispatch, SetStateAction } from "react";

export const ChatContext = createContext<{
  isRecording: boolean;
  setIsRecording: Dispatch<SetStateAction<boolean>>;
  transcriptions: string[];
  setTranscriptions: Dispatch<SetStateAction<string[]>>;
}>({
  isRecording: false,
  setIsRecording: () => {},
  transcriptions: [],
  setTranscriptions: () => {},
});
