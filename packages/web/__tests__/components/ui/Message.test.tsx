import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import Message from "@/components/ui/Message";
import { Message as MessageType } from "@/types/Message";

describe("Message", () => {
  const mockUserMessage: MessageType = {
    id: "1",
    content: "Hello, this is a user message",
    timestamp: new Date("2025-10-03T10:30:00"),
    sender: "user",
  };

  const mockAssistantMessage: MessageType = {
    id: "2",
    content: "Hello, this is an assistant response",
    timestamp: new Date("2025-10-03T10:31:00"),
    sender: "assistant",
  };

  it("renders user message with correct content and styling", () => {
    render(<Message message={mockUserMessage} />);

    const messageContent = screen.getByText("Hello, this is a user message");
    expect(messageContent).toBeInTheDocument();
    const messageContainer = messageContent.parentElement?.parentElement;
    expect(messageContainer).toHaveStyle({
      "background-color": "var(--joy-palette-primary-100, #E3EFFB)",
    });

    const timestampElement = screen.getByText(/\d{2}:\d{2}/);
    expect(timestampElement).toBeInTheDocument();
  });

  it("renders assistant message with correct content and styling", () => {
    render(<Message message={mockAssistantMessage} />);

    const messageContent = screen.getByText(
      "Hello, this is an assistant response",
    );
    expect(messageContent).toBeInTheDocument();
    const messageContainer = messageContent.parentElement?.parentElement;
    expect(messageContainer).toHaveStyle({
      "background-color": "var(--joy-palette-neutral-50, #FBFCFE)",
    });

    const timestampElement = screen.getByText(/\d{2}:\d{2}/);
    expect(timestampElement).toBeInTheDocument();
  });

  it("formats timestamp correctly in 24-hour format", () => {
    const messageWithAfternoonTime: MessageType = {
      id: "3",
      content: "Afternoon message",
      timestamp: new Date("2025-10-03T14:45:30"),
      sender: "user",
    };

    render(<Message message={messageWithAfternoonTime} />);

    const timestampElement = screen.getByText("14:45");
    expect(timestampElement).toBeInTheDocument();
  });

  it("renders message with empty content", () => {
    const emptyMessage: MessageType = {
      id: "6",
      content: "",
      timestamp: new Date("2025-10-03T10:30:00"),
      sender: "user",
    };

    render(<Message message={emptyMessage} />);

    const timestampElement = screen.getByText(/\d{2}:\d{2}/);
    expect(timestampElement).toBeInTheDocument();
  });
});
