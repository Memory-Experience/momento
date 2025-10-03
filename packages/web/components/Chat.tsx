"use client";

import { FC, useCallback, useEffect, useState } from "react";
import { Button, Textarea } from "@mui/joy";
import { Help, Save } from "@mui/icons-material";
import TranscribedRecorder from "./controls/TranscribedRecorder";
import { ChunkType, MemoryChunk } from "protos/generated/ts/stt";
import { useWebSocket } from "@/hooks/useWebSocket";
import { Message } from "@/types/Message";
import MessageList from "./ui/MessageList";

const Chat: FC = () => {
  const [mode, setMode] = useState<"memory" | "question" | null>(null);
  const url =
    mode === "memory"
      ? "ws://localhost:8080/ws/memory"
      : mode === "question"
        ? "ws://localhost:8080/ws/ask"
        : null;
  const { isConnected, connect, addEventListener, send } = useWebSocket(url);
  const [text, setText] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);

  const handleTranscription = (transcript: MemoryChunk) => {
    console.debug("Chat#handleTranscription", transcript);
    setText((prevText) => {
      return prevText + (transcript.textData || "");
    });
  };

  const saveMemory = useCallback(() => {
    connect();

    addEventListener("open", () => {
      send(
        MemoryChunk.encode({
          textData: text,
          metadata: {
            memoryId: crypto.randomUUID(),
            sessionId: "",
            type: ChunkType.MEMORY,
            isFinal: true,
            score: 0,
          },
        }).finish(),
      );
    });

    addEventListener("message", async (e: MessageEvent) => {
      const data = e.data instanceof Blob ? await e.data.bytes() : e.data;
      if (data) {
        const message = MemoryChunk.decode(new Uint8Array(data));
        if (
          message.metadata?.type === ChunkType.MEMORY &&
          message.metadata?.isFinal
        ) {
          setMessages((prev) => [
            ...prev,
            {
              id: crypto.randomUUID(),
              isFinal: true,
              content: text,
              timestamp: new Date(),
              sender: "user",
            },
            {
              id: message.metadata?.memoryId ?? crypto.randomUUID(),
              isFinal: message.metadata?.isFinal ?? false,
              content: message.textData || "",
              timestamp: new Date(),
              sender: "assistant",
            },
          ]);
          setText("");
        }
      } else {
        console.warn("Chat: Received empty message", data);
      }
    });
  }, [addEventListener, connect, send, text]);

  const askQuestion = useCallback(() => {
    connect();

    addEventListener("open", () => {
      send(
        MemoryChunk.encode({
          textData: text,
          metadata: {
            memoryId: "",
            sessionId: crypto.randomUUID(),
            type: ChunkType.QUESTION,
            isFinal: true,
            score: 0,
          },
        }).finish(),
      );
    });

    addEventListener("message", async (e: MessageEvent) => {
      const data = e.data instanceof Blob ? await e.data.bytes() : e.data;
      if (data) {
        const message = MemoryChunk.decode(new Uint8Array(data));
        console.debug("Chat: Received message", message);
        if (message.metadata?.type === ChunkType.ANSWER) {
          setMessages((prev) => {
            const sessionId = message.metadata?.sessionId;
            const messageIx = prev.findIndex(({ id }) => id === sessionId);

            if (messageIx >= 0) {
              // update existing message
              const post = [...prev];
              post[messageIx] = {
                ...post[messageIx],
                isFinal:
                  message.metadata?.isFinal ?? post[messageIx].isFinal ?? false,
                content: post[messageIx].content + (message.textData ?? ""),
              };
              return post;
            }
            // create new message pair (question + answer)
            return [
              ...prev,
              {
                id: crypto.randomUUID(),
                isFinal: true,
                content: text,
                timestamp: new Date(),
                sender: "user",
              },
              {
                id: sessionId ?? crypto.randomUUID(),
                isFinal: message.metadata?.isFinal ?? false,
                content: message.textData ?? "",
                timestamp: new Date(),
                sender: "assistant",
              },
            ];
          });

          if (message.metadata?.isFinal) {
            setText("");
          }
        }
      } else {
        console.warn("Chat: Received empty message", data);
      }
    });
  }, [addEventListener, connect, send, text]);

  useEffect(() => {
    if (!mode) {
      return;
    }

    if (mode === "memory") {
      saveMemory();
    } else if (mode === "question") {
      askQuestion();
    }

    setMode(null);
  }, [mode, setMode, askQuestion, saveMemory]);

  return (
    <div className="h-full grid grid-rows-[1fr_auto] gap-4">
      <div className="overflow-y-auto min-h-0">
        <MessageList messages={messages} />
      </div>

      <div className="w-full flex items-start gap-2">
        <Textarea
          minRows={1}
          maxRows={4}
          sx={{
            width: "75%",
            margin: "0 auto",
          }}
          placeholder="Type your memory/question..."
          disabled={isConnected}
          value={text}
          onChange={(e) => setText(e.target.value)}
          endDecorator={
            <div className="w-full flex gap-2">
              <TranscribedRecorder onTranscription={handleTranscription} />
              <Button
                sx={{ ml: "auto" }}
                variant="plain"
                color="primary"
                size="sm"
                startDecorator={<Help />}
                onClick={() => setMode("question")}
                disabled={!text.trim() || isConnected}
              >
                Ask Question
              </Button>
              <Button
                variant="plain"
                color="danger"
                size="sm"
                startDecorator={<Save />}
                onClick={() => setMode("memory")}
                disabled={!text.trim() || isConnected}
              >
                Save Memory
              </Button>
            </div>
          }
        />
      </div>
    </div>
  );
};

export default Chat;
