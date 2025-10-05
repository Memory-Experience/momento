import "@testing-library/jest-dom";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  act,
} from "@testing-library/react";
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
    const messageContainer = messageContent.parentElement;
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
    const messageContainer = messageContent.parentElement;
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

  it("renders thinking indicator when isThinking is true without thinking time", () => {
    const thinkingMessage: MessageType = {
      id: "7",
      content: "Processing your request",
      timestamp: new Date("2025-10-03T10:30:00"),
      sender: "assistant",
      isThinking: true,
    };

    render(<Message message={thinkingMessage} />);

    const header = screen.getByRole("button", { name: "Thinking..." });
    expect(header).toBeInTheDocument();
    expect(header).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByRole("progressbar")).toBeInTheDocument();
  });

  it("renders thinking time when message is final and thinking was completed", async () => {
    jest.useFakeTimers();

    const { rerender } = render(
      <Message
        message={{
          id: "8",
          content: "Response",
          timestamp: new Date("2025-10-03T10:30:00"),
          sender: "assistant",
          isThinking: true,
        }}
      />,
    );

    const header = screen.getByRole("button", { name: "Thinking..." });
    expect(header).toBeInTheDocument();
    expect(header).toHaveAttribute("aria-expanded", "false");

    act(() => {
      jest.advanceTimersByTime(2157); // 2.157s
    });

    act(() => {
      rerender(
        <Message
          message={{
            id: "8",
            content: "Response",
            timestamp: new Date("2025-10-03T10:30:00"),
            sender: "assistant",
            isThinking: false,
            thinkingText: "Foobar",
            isFinal: true,
          }}
        />,
      );
    });

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: "Thought for 2.1s" }),
      ).toBeInTheDocument();
    });

    const accordionContent = screen.getByText("Foobar");
    expect(accordionContent).not.toBeVisible();

    fireEvent.click(header);

    expect(header).toHaveAttribute("aria-expanded", "true");
    expect(accordionContent).toBeVisible();

    jest.useRealTimers();
  });

  it("does not render thinking indicator when isThinking is false and no thinkingText is provided", () => {
    const normalMessage: MessageType = {
      id: "11",
      content: "Regular message",
      timestamp: new Date("2025-10-03T10:30:00"),
      sender: "assistant",
      isThinking: false,
    };

    render(<Message message={normalMessage} />);

    expect(screen.queryByText("Thinking...")).not.toBeInTheDocument();
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument();
  });
});
