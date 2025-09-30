"use client";

import { FC, useState } from "react";
import { Button, Textarea } from "@mui/joy";
import { Help, Save } from "@mui/icons-material";
import TranscribedRecorder from "./controls/TranscribedRecorder";
import { ChunkType, MemoryChunk } from "protos/generated/ts/stt";
import { useWebSocket } from "@/hooks/useWebSocket";

const Chat: FC = () => {
  const { isConnected, connect, addEventListener, send } = useWebSocket(
    "ws://localhost:8080/ws/memory",
  );
  const [text, setText] = useState("");
  const [messages, setMessages] = useState<string[]>([]);

  const handleTranscription = (transcript: MemoryChunk) => {
    console.debug("Chat#handleTranscription", transcript);
    setText((prevText) => {
      return prevText + (transcript.textData || "");
    });
  };

  const saveMemory = () => {
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
          setMessages((prev) => [...prev, text, message.textData || ""]);
          setText("");
        }
      } else {
        console.warn("Chat: Received empty message", data);
      }
    });
  };

  return (
    <div className="w-full h-full flex flex-col gap-2">
      <div className="flex-1 overflow-y-auto">
        <div className="text-gray-500 text-center">Messages section</div>
        {messages.map((msg, idx) => (
          <div key={idx} className="p-2 border-b">
            {msg}
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
                  disabled={!text.trim() || isConnected}
                >
                  Ask Question
                </Button>
                <Button
                  variant="plain"
                  color="danger"
                  size="sm"
                  startDecorator={<Save />}
                  onClick={saveMemory}
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
