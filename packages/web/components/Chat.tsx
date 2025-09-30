"use client";

import { FC, useCallback, useEffect, useState } from "react";
import { Button, Textarea } from "@mui/joy";
import { Help, Save } from "@mui/icons-material";
import TranscribedRecorder from "./controls/TranscribedRecorder";
import { ChunkType, MemoryChunk } from "protos/generated/ts/stt";
import { useWebSocket } from "@/hooks/useWebSocket";

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
  const [messages, setMessages] = useState<{ id: string; text: string }[]>([]);

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
            { id: crypto.randomUUID(), text },
            {
              id: message.metadata?.memoryId ?? crypto.randomUUID(),
              text: message.textData || "",
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
            const messageIx = prev.findIndex(
              ({ id }) => id === message.metadata?.sessionId,
            );
            if (messageIx >= 0) {
              prev[messageIx].text += message.textData ?? "";
              console.debug("Chat: update existing message for question", [
                ...prev,
              ]);
              return [...prev];
            }
            const post = [
              ...prev,
              { id: crypto.randomUUID(), text },
              {
                id: message.metadata?.sessionId ?? crypto.randomUUID(),
                text: message.textData ?? "",
              },
            ];
            console.debug("Chat: update new message for question", [...post]);
            return post;
          });
          setText("");
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

    addEventListener("close", () => {
      console.log("Chat: WebSocket closed");
      setMode(null);
    });
  }, [addEventListener, connect, askQuestion, saveMemory, mode]);

  return (
    <div className="w-full h-full flex flex-col gap-2">
      <div className="flex-1 overflow-y-auto">
        <div className="text-gray-500 text-center">Messages section</div>
        {messages.map(({ id, text }) => (
          <div key={id} className="p-2 border-b">
            {text}
          </div>
        ))}
      </div>
      <div className="flex-shrink-0">
        <div className="w-full flex items-start gap-2">
          <Textarea
            minRows={1}
            maxRows={4}
            sx={{ width: "75%", margin: "0 auto" }}
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
    </div>
  );
};

export default Chat;
