import "@testing-library/jest-dom";
import { render } from "@testing-library/react";
import MessageList from "@/components/ui/MessageList";
import { Message as MessageType } from "@/types/Message";
import MessageEmptyState from "@/components/ui/MessageEmptyState";
import Message from "@/components/ui/Message";

jest.mock("./MessageEmptyState", () => jest.fn(() => <></>));
jest.mock("./Message", () => jest.fn(() => <></>));

beforeEach(() => {
  jest.clearAllMocks();
});

describe("MessageList", () => {
  const mockMessages: MessageType[] = [
    {
      id: "1",
      content: "First user message",
      timestamp: new Date("2025-10-03T10:30:00"),
      sender: "user",
    },
    {
      id: "2",
      content: "Assistant response",
      timestamp: new Date("2025-10-03T10:31:00"),
      sender: "assistant",
    },
    {
      id: "3",
      content: "Second user message",
      timestamp: new Date("2025-10-03T10:32:00"),
      sender: "user",
    },
  ];

  it("renders empty state when no messages are provided", () => {
    expect(Message).not.toHaveBeenCalled();
    expect(MessageEmptyState).not.toHaveBeenCalled();

    render(<MessageList messages={[]} />);

    expect(Message).not.toHaveBeenCalled();
    expect(MessageEmptyState).toHaveBeenCalledTimes(1);
    expect(MessageEmptyState).toHaveBeenCalledWith({}, undefined);
  });

  it("renders messages when messages are provided", () => {
    expect(Message).not.toHaveBeenCalled();
    expect(MessageEmptyState).not.toHaveBeenCalled();

    render(<MessageList messages={mockMessages} />);

    expect(MessageEmptyState).not.toHaveBeenCalled();
    expect(Message).toHaveBeenCalledTimes(3);
    expect(Message).toHaveBeenNthCalledWith(
      1,
      { message: mockMessages[0] },
      undefined,
    );
    expect(Message).toHaveBeenNthCalledWith(
      2,
      { message: mockMessages[1] },
      undefined,
    );
    expect(Message).toHaveBeenNthCalledWith(
      3,
      { message: mockMessages[2] },
      undefined,
    );
  });

  it("scrolls to the bottom after rerender", () => {
    const expectedScrollParams = { behavior: "smooth" };

    const { rerender, container } = render(
      <MessageList messages={[mockMessages[1]]} />,
    );

    const messagesEndRef = container.querySelector("div > div > div > div");
    expect(messagesEndRef).toBeInTheDocument();
    expect(messagesEndRef?.childNodes).toHaveLength(0);
    expect(messagesEndRef?.scrollIntoView).toHaveBeenCalledTimes(1);
    expect(messagesEndRef?.scrollIntoView).toHaveBeenNthCalledWith(
      1,
      expectedScrollParams,
    );

    rerender(<MessageList messages={mockMessages} />);

    expect(messagesEndRef).toBeInTheDocument();
    expect(messagesEndRef?.scrollIntoView).toHaveBeenCalledTimes(2);
    expect(messagesEndRef?.scrollIntoView).toHaveBeenNthCalledWith(
      2,
      expectedScrollParams,
    );
  });

  it("switches between empty state and message list correctly", () => {
    expect(Message).not.toHaveBeenCalled();
    expect(MessageEmptyState).not.toHaveBeenCalled();

    const { rerender } = render(<MessageList messages={[]} />);

    expect(Message).not.toHaveBeenCalled();
    expect(MessageEmptyState).toHaveBeenCalledTimes(1);

    rerender(<MessageList messages={mockMessages} />);

    expect(Message).toHaveBeenCalledTimes(3);
    expect(MessageEmptyState).toHaveBeenCalledTimes(1);

    rerender(<MessageList messages={[]} />);

    expect(Message).toHaveBeenCalledTimes(3);
    expect(MessageEmptyState).toHaveBeenCalledTimes(2);
  });
});
