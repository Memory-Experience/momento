import { cn } from "@/utils";
import { motion } from "motion/react";
import { MemoryChunk } from "protos/generated/ts/stt";
import { FC, useEffect, useState } from "react";

interface TranscriptionMessageProps {
  chunk: MemoryChunk;
  className?: string;
  position?: "left" | "right";
  header?: React.ReactNode;
}

const TranscriptionMessage: FC<TranscriptionMessageProps> = ({
  chunk,
  className,
  position = "right",
  header,
}) => {
  const [text, setText] = useState<string>(chunk.textData || "");

  // Update text when chunk changes
  useEffect(() => {
    if (chunk.textData) {
      setText((prev) => {
        // If the new text is longer and starts with the old text,
        // it's an update/delta to the existing transcription
        if (chunk.textData!.startsWith(prev)) {
          return chunk.textData!;
        }
        // If it's completely different text, replace it
        return chunk.textData!;
      });
    }
  }, [chunk.textData]);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn(
        "w-full p-4 rounded-xl border",
        position === "right" ? "ml-auto bg-card border-border" : "mr-auto",
        className,
      )}
    >
      {header && header}
      <div className="whitespace-pre-wrap">{text}</div>
    </motion.div>
  );
};

export default TranscriptionMessage;
