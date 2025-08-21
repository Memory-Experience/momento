"use client";
import { cn } from "@/utils";
import { AnimatePresence, motion } from "motion/react";
import { ComponentRef, forwardRef } from "react";
import { Mic } from "lucide-react";
import { TranscriptionItem } from "@/context/ChatContext";

interface MessagesProps {
  transcriptions: TranscriptionItem[];
}

const Messages = forwardRef<ComponentRef<typeof motion.div>, MessagesProps>(
  function Messages({ transcriptions }, ref) {
    const messages = transcriptions.reduce((acc, curr) => {
      const lastMessage = acc[acc.length - 1];
      if (lastMessage && lastMessage.type === curr.type) {
        lastMessage.text += curr.text;
      } else {
        acc.push({ ...curr });
      }
      return acc;
    }, [] as TranscriptionItem[]);
    console.debug("Rendering Messages component", transcriptions, messages);
    return (
      <motion.div
        layoutScroll
        className={"grow overflow-auto p-4 pt-18"}
        ref={ref}
      >
        <motion.div className={"max-w-2xl mx-auto w-full flex flex-col gap-4"}>
          {transcriptions.length === 0 ? (
            <div className="text-center py-12">
              <div className="flex justify-center mb-4">
                <div className="p-4 bg-muted rounded-full">
                  <Mic className="h-8 w-8 text-muted-foreground" />
                </div>
              </div>
              <h3 className="text-lg font-medium text-muted-foreground mb-2">
                Connect to start transcribing
              </h3>
              <p className="text-sm text-muted-foreground">
                Connect to start recording with real-time transcription
              </p>
            </div>
          ) : (
            <AnimatePresence mode={"popLayout"}>
              {messages.map((item, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={cn(
                    "w-full p-4 rounded-xl border",
                    item.type === "transcript"
                      ? "ml-auto bg-card border-border"
                      : "mr-auto bg-primary/10 border-primary/20",
                  )}
                >
                  {item.type === "answer" && (
                    <div className="text-sm font-medium text-primary mb-2">
                      Answer:
                    </div>
                  )}
                  <div className="whitespace-pre-wrap">{item.text}</div>
                </motion.div>
              ))}
            </AnimatePresence>
          )}
        </motion.div>
      </motion.div>
    );
  },
);

export default Messages;
