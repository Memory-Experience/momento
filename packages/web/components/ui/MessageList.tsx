"use client";

import { FC, useEffect, useRef } from "react";
import { Box } from "@mui/joy";
import { Message } from "@/types/Message";
import MessageComponent from "./Message";
import MessageEmptyState from "./MessageEmptyState";

interface MessageListProps {
  messages: Message[];
}

const MessageList: FC<MessageListProps> = ({ messages }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  return (
    <Box
      sx={{
        height: "100%",
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      {messages.length === 0 ? (
        <MessageEmptyState />
      ) : (
        <Box
          sx={{
            flex: 1,
            overflowY: "auto",
            p: 2,
            minHeight: 0,
          }}
        >
          {messages.map((message) => (
            <MessageComponent key={message.id} message={message} />
          ))}
          <div ref={messagesEndRef} />
        </Box>
      )}
    </Box>
  );
};

export default MessageList;
