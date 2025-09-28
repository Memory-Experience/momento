"use client";

import { FC, useState } from "react";
import { Button, Textarea } from "@mui/joy";
import { Help, Save } from "@mui/icons-material";
import TranscribedRecorder from "./controls/TranscribedRecorder";
import { MemoryChunk } from "protos/generated/ts/stt";

const Chat: FC = () => {
  const [text, setText] = useState("");

  const handleTranscription = (transcript: MemoryChunk) => {
    console.debug("Chat#handleTranscription", transcript);
    setText((prevText) => {
      return prevText + (transcript.textData || "");
    });
  };

  return (
    <div className="w-full h-full flex flex-col gap-2">
      <div className="flex-1 overflow-y-auto">
        <div className="text-gray-500 text-center">Messages section</div>
        {/* This is where your messages will go */}
      </div>
      <div className="flex-shrink-0">
        <div className="w-full flex items-start gap-2">
          <Textarea
            minRows={1}
            maxRows={4}
            sx={{ width: "75%", margin: "0 auto" }}
            placeholder="Type your memory/question..."
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
                >
                  Ask Question
                </Button>
                <Button
                  variant="plain"
                  color="danger"
                  size="sm"
                  startDecorator={<Save />}
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
