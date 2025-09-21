"use client";

import { FC } from "react";
import { Button, Textarea } from "@mui/joy";
import { Help, MicNone, Save } from "@mui/icons-material";

const Chat: FC = () => {
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
            endDecorator={
              <div className="w-full flex gap-2">
                <Button
                  variant="plain"
                  color="neutral"
                  size="sm"
                  startDecorator={<MicNone />}
                >
                  Dictate
                </Button>
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
