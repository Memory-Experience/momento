"use client";
import { AnimatePresence, motion } from "motion/react";
import { ComponentRef, forwardRef, ReactNode, useEffect, useRef } from "react";
import { Mic } from "lucide-react";

interface MessagesProps {
  messages: ReactNode[];
  mode?: "memory" | "question";
}

const Messages = forwardRef<ComponentRef<typeof motion.div>, MessagesProps>(
  function Messages({ messages, mode }, ref) {
    // Debug: Log when messages change
    const prevMessagesLength = useRef(messages.length);

    useEffect(() => {
      if (messages.length !== prevMessagesLength.current) {
        console.log(
          `Messages changed: ${prevMessagesLength.current} -> ${messages.length}`,
        );
        prevMessagesLength.current = messages.length;
      }
    }, [messages]);

    return (
      <motion.div
        layoutScroll
        className={"grow overflow-auto p-4 pt-18"}
        ref={ref}
      >
        <motion.div className={"max-w-2xl mx-auto w-full flex flex-col gap-4"}>
          {messages.length === 0 ? (
            <div className="text-center py-12">
              <div className="flex justify-center mb-4">
                <div className="p-4 bg-muted rounded-full">
                  <Mic className="h-8 w-8 text-muted-foreground" />
                </div>
              </div>
              <h3 className="text-lg font-medium text-muted-foreground mb-2">
                {mode
                  ? mode === "memory"
                    ? "Recording a new memory..."
                    : "Ask your question..."
                  : "Ready to record"}
              </h3>
              <p className="text-sm text-muted-foreground">
                {mode
                  ? mode === "memory"
                    ? "Your voice will be transcribed and saved"
                    : "Your question will be answered based on your memories"
                  : "Use the buttons below to record a memory or ask a question"}
              </p>
            </div>
          ) : (
            <AnimatePresence mode={"popLayout"}>
              {messages.map((message, index) => (
                <div key={`message-wrapper-${index}`}>{message}</div>
              ))}
            </AnimatePresence>
          )}
        </motion.div>
      </motion.div>
    );
  },
);

export default Messages;
