import { FC } from "react";
import { MemoryChunk } from "protos/generated/ts/stt";
import TranscriptionMessage from "./TranscriptionMessage";

interface MemoryMessageProps {
  chunk: MemoryChunk;
}

const MemoryMessage: FC<MemoryMessageProps> = ({ chunk }) => {
  const header = (
    <div className="text-xs font-medium text-muted-foreground mb-2 flex items-center gap-1">
      <div className="w-1.5 h-1.5 bg-muted-foreground/50 rounded-full"></div>
      Related Memory:
    </div>
  );

  return (
    <TranscriptionMessage
      chunk={chunk}
      position="left"
      header={header}
      className="bg-muted/50 border-muted-foreground/20"
    />
  );
};

export default MemoryMessage;
