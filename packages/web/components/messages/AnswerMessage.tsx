import { FC } from "react";
import { MemoryChunk } from "protos/generated/ts/stt";
import TranscriptionMessage from "./TranscriptionMessage";

interface AnswerMessageProps {
  chunk: MemoryChunk;
}

const AnswerMessage: FC<AnswerMessageProps> = ({ chunk }) => {
  const header = (
    <div className="text-sm font-medium text-primary mb-2">Answer:</div>
  );

  return (
    <TranscriptionMessage
      chunk={chunk}
      position="left"
      header={header}
      className="bg-primary/10 border-primary/20"
    />
  );
};

export default AnswerMessage;
