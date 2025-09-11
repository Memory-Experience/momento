import { useState, forwardRef, useImperativeHandle, useEffect } from "react";
import { MemoryChunk } from "protos/generated/ts/stt";
import { cn } from "@/utils";
import { motion } from "motion/react";
import { Card } from "@/components/ui/card";

export interface StreamingMessageHandle {
  updateContent: (newText: string) => void;
  appendContent: (additionalText: string) => void;
  processChunk: (chunk: MemoryChunk) => void;
  markComplete: () => void;
}

interface StreamingMessageProps {
  className?: string;
  initialContent?: string;
  onComplete?: () => void;
}

const StreamingMessage = forwardRef<
  StreamingMessageHandle,
  StreamingMessageProps
>(function StreamingMessage(
  { className, initialContent = "", onComplete },
  ref,
) {
  const [content, setContent] = useState<string>(initialContent);
  const [isComplete, setIsComplete] = useState<boolean>(false);

  useEffect(() => {
    console.log("StreamingMessage mounted");
    return () => {
      console.log("StreamingMessage unmounted");
    };
  }, []);

  // Call onComplete when the message is marked as complete
  useEffect(() => {
    if (isComplete && onComplete) {
      onComplete();
    }
  }, [isComplete, onComplete]);

  // Expose methods to update the content
  useImperativeHandle(
    ref,
    () => ({
      updateContent: (newText: string) => {
        console.log("updateContent called with:", newText);
        setContent(newText);
      },
      appendContent: (additionalText: string) => {
        console.log("appendContent called with:", additionalText);
        setContent((prev) => prev + additionalText);
      },
      processChunk: (chunk: MemoryChunk) => {
        if (chunk.textData) {
          console.log("processChunk called with:", chunk.textData);
          setContent((prev) => prev + chunk.textData);
        }

        // Check if this is the final chunk
        if (chunk.metadata?.isFinal) {
          console.log("Final chunk received, marking message as complete");
          setIsComplete(true);
        }
      },
      markComplete: () => {
        console.log("Message manually marked as complete");
        setIsComplete(true);
      },
    }),
    [],
  );

  // Always render the component if we have content or it's marked as complete
  return content || isComplete ? (
    <motion.div
      layout
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className={cn("text-sm", className)}>
        <div className="whitespace-pre-wrap">
          {content || "Waiting for content..."}
        </div>
      </Card>
    </motion.div>
  ) : (
    <div />
  );
});

export default StreamingMessage;
