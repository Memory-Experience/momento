import { useState, useImperativeHandle, forwardRef, useEffect } from "react";
import { MemoryChunk } from "protos/generated/ts/stt";
import StreamingMessage, { StreamingMessageHandle } from "./StreamingMessage";
import { useRef } from "react";

export interface AnswerMessageHandle {
  updateContent: (newText: string) => void;
  appendContent: (additionalText: string) => void;
  processChunk: (chunk: MemoryChunk) => void;
  markComplete: () => void;
}

interface AnswerMessageProps {
  initialContent?: string;
  onComplete?: () => void;
}

const AnswerMessage = forwardRef<AnswerMessageHandle, AnswerMessageProps>(
  function AnswerMessage({ initialContent = "", onComplete }, ref) {
    const [content, setContent] = useState<string>(initialContent);
    const [isComplete, setIsComplete] = useState<boolean>(false);
    const streamMessageRef = useRef<StreamingMessageHandle>(null);

    // Call onComplete when the message is marked as complete
    useEffect(() => {
      if (isComplete && onComplete) {
        onComplete();
      }
    }, [isComplete, onComplete]);

    // Expose methods to update the answer content
    useImperativeHandle(ref, () => ({
      updateContent: (newText: string) => {
        setContent(newText);
        if (streamMessageRef.current) {
          streamMessageRef.current.updateContent(newText);
        }
      },
      appendContent: (additionalText: string) => {
        setContent((prev) => prev + additionalText);
        if (streamMessageRef.current) {
          streamMessageRef.current.appendContent(additionalText);
        }
      },
      processChunk: (chunk: MemoryChunk) => {
        if (chunk.textData) {
          setContent((prev) => prev + chunk.textData);
          if (streamMessageRef.current) {
            streamMessageRef.current.appendContent(chunk.textData);
          }
        }

        // Check if this is the final chunk
        if (chunk.metadata?.isFinal) {
          console.log(
            "Final answer chunk received, marking answer as complete",
          );
          setIsComplete(true);
          if (streamMessageRef.current) {
            streamMessageRef.current.markComplete();
          }
        }
      },
      markComplete: () => {
        setIsComplete(true);
        if (streamMessageRef.current) {
          streamMessageRef.current.markComplete();
        }
      },
    }));

    return (
      <div>
        <StreamingMessage
          ref={streamMessageRef}
          initialContent={content}
          className=""
          onComplete={() => {
            if (onComplete) onComplete();
          }}
        />
      </div>
    );
  },
);

export default AnswerMessage;
