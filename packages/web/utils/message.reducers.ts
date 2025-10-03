import { Message } from "@/types/Message";
import { MemoryChunk, ChunkType } from "protos/generated/ts/stt";

enum MESSAGE_MARKER {
  THINKING_START = "<think>",
  THINKING_STOP = "</think>",
}

export const reduceQuestionMessages = (
  prevMessages: Message[],
  currMessage: MemoryChunk,
): Message[] => {
  if (currMessage.metadata?.type === ChunkType.QUESTION) {
    return [
      ...prevMessages,
      {
        id: currMessage.metadata.sessionId ?? crypto.randomUUID(),
        isFinal: true,
        content: currMessage.textData ?? "",
        timestamp: new Date(),
        sender: "user",
      },
    ];
  }
  const sessionId = currMessage.metadata?.sessionId;
  const lastMessageIx = prevMessages.findIndex(
    ({ id, sender }) => id === sessionId && sender === "assistant",
  );
  const lastMessage =
    lastMessageIx > -1 ? prevMessages[lastMessageIx] : undefined;

  const textData = currMessage.textData;

  const thinkingStartIx =
    textData?.indexOf(MESSAGE_MARKER.THINKING_START) ?? -1;
  const thinkingStopIx = textData?.indexOf(MESSAGE_MARKER.THINKING_STOP) ?? -1;

  const isThinking = lastMessage?.isThinking || thinkingStartIx > -1;

  const thinkingText = isThinking
    ? textData?.substring(
        Math.max(
          thinkingStartIx > -1
            ? thinkingStartIx + MESSAGE_MARKER.THINKING_START.length
            : -1,
          0,
        ),
        thinkingStopIx > -1 ? thinkingStopIx : textData?.length,
      )
    : undefined;
  const beforeThinking = textData?.substring(
    0,
    thinkingStartIx > -1 ? thinkingStartIx : 0,
  );
  const afterThinking = textData?.substring(
    thinkingStopIx > -1
      ? thinkingStopIx + MESSAGE_MARKER.THINKING_STOP.length
      : textData?.length,
  );
  const content = isThinking
    ? (beforeThinking ?? "") + (afterThinking ?? "")
    : textData;

  console.debug(
    "Message.reducers",
    { ...currMessage },
    isThinking,
    thinkingText,
    content,
    `Before: ${beforeThinking}, After: ${afterThinking}`,
  );

  if (lastMessage) {
    const message = {
      ...lastMessage,
      content: lastMessage.content + content,
      isFinal: currMessage.metadata?.isFinal ?? lastMessage.isFinal ?? false,
      isThinking: thinkingStopIx > -1 ? false : isThinking,
      ...(thinkingText
        ? { thinkingText: (lastMessage.thinkingText ?? "") + thinkingText }
        : {}),
    };
    return [
      ...prevMessages.slice(0, lastMessageIx),
      message,
      ...prevMessages.slice(lastMessageIx + 1),
    ];
  }

  return [
    ...prevMessages,
    {
      id: sessionId ?? crypto.randomUUID(),
      isFinal: currMessage.metadata?.isFinal ?? false,
      isThinking: thinkingStopIx > -1 ? false : isThinking,
      thinkingText: thinkingText,
      content: content ?? "",
      timestamp: new Date(),
      sender: "assistant",
    },
  ];
};
