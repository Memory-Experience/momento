export type MessageSender = "user" | "assistant";

export interface Message {
  id: string;
  content: string;
  timestamp: Date;
  sender: MessageSender;
  isFinal?: boolean;
  isThinking?: boolean;
  thinkingText?: string;
}
