"use client";

import { FC } from "react";
import { Box, Card, Typography } from "@mui/joy";
import { Message as MessageType } from "@/types/Message";

interface MessageProps {
  message: MessageType;
}

const Message: FC<MessageProps> = ({ message }) => {
  const isUser = message.sender === "user";

  const formatTimestamp = (timestamp: Date) => {
    return new Intl.DateTimeFormat("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(timestamp);
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
        <Box sx={{ p: 1.5 }}>
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
        </Box>
      </Card>
    </Box>
  );
};

export default Message;
