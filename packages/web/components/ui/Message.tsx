"use client";

import { FC, useState, useEffect } from "react";
import {
  Box,
  Card,
  Typography,
  CircularProgress,
  Accordion,
  AccordionDetails,
  AccordionSummary,
} from "@mui/joy";
import { Message as MessageType } from "@/types/Message";

interface MessageProps {
  message: MessageType;
}

const Message: FC<MessageProps> = ({ message }) => {
  const isUser = message.sender === "user";
  const [thinkingTime, setThinkingTime] = useState(0);
  const [startTime, setStartTime] = useState<number | null>(null);

  // Timer for thinking indicator
  useEffect(() => {
    let interval: NodeJS.Timeout;

    if (message.isThinking) {
      if (!startTime) {
        setStartTime(Date.now());
      }
      interval = setInterval(() => {
        if (startTime) {
          setThinkingTime((Date.now() - startTime) / 1000);
        }
      }, 100);
    }

    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [message.isThinking, startTime]);

  // Reset for new thinking session
  useEffect(() => {
    if (message.isThinking && !startTime) {
      setStartTime(Date.now());
      setThinkingTime(0);
    }
  }, [message.isThinking, startTime]);

  const formatTimestamp = (timestamp: Date) => {
    return new Intl.DateTimeFormat("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(timestamp);
  };

  const formatThinkingTime = (time: number) => {
    return `${time.toFixed(1)}s`;
  };

  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: isUser ? "flex-end" : "flex-start",
        mb: 2,
      }}
    >
      <Card
        variant="outlined"
        sx={{
          maxWidth: "70%",
          minWidth: "20%",
          backgroundColor: isUser ? "primary.100" : "neutral.50",
          borderColor: isUser ? "primary.300" : "neutral.200",
          position: "relative",
        }}
      >
        {(message.thinkingText || message.isThinking) && (
          <Accordion>
            <AccordionSummary>
              <Typography level="body-sm">
                {message.isThinking ? (
                  <>
                    <CircularProgress size="sm" /> Thinking...
                  </>
                ) : (
                  <>Thought for {formatThinkingTime(thinkingTime)}</>
                )}
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Typography
                level="body-sm"
                sx={{
                  whiteSpace: "pre-wrap",
                }}
              >
                {message.thinkingText?.trim()}
              </Typography>
            </AccordionDetails>
          </Accordion>
        )}

        <Typography
          level="body-md"
          sx={{
            color: isUser ? "primary.800" : "neutral.800",
            whiteSpace: "pre-wrap",
            wordBreak: "break-word",
          }}
        >
          {message.content}
        </Typography>

        <Typography
          level="body-xs"
          sx={{
            color: isUser ? "primary.600" : "neutral.600",
            mt: 0.5,
            textAlign: "right",
          }}
        >
          {formatTimestamp(message.timestamp)}
        </Typography>
      </Card>
    </Box>
  );
};

export default Message;
