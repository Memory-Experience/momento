"use client";
import { cn } from "@/utils";
import { AnimatePresence, motion } from "motion/react";
import { ComponentRef, forwardRef } from "react";
import { Mic } from "lucide-react";

interface MessagesProps {
  transcriptions: string[];
}

const Messages = forwardRef<ComponentRef<typeof motion.div>, MessagesProps>(
  function Messages({ transcriptions }, ref) {
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
              <div
                className={cn(
                  "w-full ml-auto bg-card",
                  "border border-border rounded-xl p-4",
                )}
              >
                {transcriptions.join(" ")}
              </div>
            </AnimatePresence>
          )}
        </motion.div>
      </motion.div>
    );
  },
);

export default Messages;
