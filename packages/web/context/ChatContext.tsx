import { createContext, Dispatch, SetStateAction } from "react";

export interface TranscriptionItem {
  type: "transcript" | "answer";
  text: string;
  timestamp: number;
}

export const ChatContext = createContext<{
  mode: "memory" | "question" | undefined;
  setMode: Dispatch<SetStateAction<"memory" | "question" | undefined>>;
  isRecording: boolean;
  setIsRecording: Dispatch<SetStateAction<boolean>>;
  transcriptions: TranscriptionItem[];
  setTranscriptions: Dispatch<SetStateAction<TranscriptionItem[]>>;
}>({
  mode: undefined,
  setMode: () => {},
  isRecording: false,
  setIsRecording: () => {},
  transcriptions: [],
  setTranscriptions: () => {},
});
