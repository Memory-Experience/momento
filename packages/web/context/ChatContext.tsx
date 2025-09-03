import { createContext, Dispatch, ReactNode, SetStateAction } from "react";

interface ChatContextType {
  mode: "memory" | "question" | undefined;
  setMode: Dispatch<SetStateAction<"memory" | "question" | undefined>>;
  isRecording: boolean;
  setIsRecording: Dispatch<SetStateAction<boolean>>;
  isProcessing: boolean;
  setIsProcessing: Dispatch<SetStateAction<boolean>>;
  messages: ReactNode[];
  setMessages: Dispatch<SetStateAction<ReactNode[]>>;
}

// Create context with default empty values
export const ChatContext = createContext<ChatContextType>({
  mode: undefined,
  setMode: () => {},
  isRecording: false,
  setIsRecording: () => {},
  isProcessing: false,
  setIsProcessing: () => {},
  messages: [],
  setMessages: () => {},
});
