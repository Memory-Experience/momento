import "@testing-library/jest-dom";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import Chat from "@/components/Chat";
import { ChunkMetadata, ChunkType, MemoryChunk } from "protos/generated/ts/stt";
import MessageList from "@/components/ui/MessageList";

jest.mock("../../components/ui/MessageList", () => jest.fn(() => <></>));

const states = {
  OPEN: WebSocket.OPEN.valueOf(),
  CLOSED: WebSocket.CLOSED.valueOf(),
};

// Simple WebSocket mock focusing on the API interaction
const mockWebSocket = {
  send: jest.fn(),
  close: jest.fn(),
  addEventListener: jest.fn(),
  readyState: states.OPEN,
};

const mockWebSocketConstructor = jest.fn(() => mockWebSocket);

// Helper to trigger WebSocket events
const triggerWebSocketEvent = (eventType: string, data?: Uint8Array) => {
  const calls = mockWebSocket.addEventListener.mock.calls;
  const handler: EventListener[] = calls
    .filter((call) => call[0] === eventType)
    .map((call) => call[1]);
  if (handler) {
    if (eventType === "message") {
      handler.forEach((curr) => {
        curr({ data } as MessageEvent);
      });
    } else if (eventType === "open") {
      mockWebSocket.readyState = states.OPEN;
      handler.forEach((curr) => curr({} as Event));
    } else if (eventType === "close") {
      mockWebSocket.readyState = states.CLOSED;
      handler.forEach((curr) => curr({} as Event));
    } else {
      handler.forEach((curr) => curr({} as Event));
    }
  }
};

let firstDate: Date | undefined;
let secondDate: Date | undefined;

let originalWebSocket: typeof WebSocket;
beforeAll(() => {
  originalWebSocket = global.WebSocket;
  // Mocking WebSocket contructor returning the mockWebSocket instance
  (global as { WebSocket: Partial<typeof WebSocket> }).WebSocket =
    mockWebSocketConstructor;
  for (const [key, value] of Object.entries(states)) {
    (global.WebSocket as unknown as Record<string, unknown>)[key] = value;
  }
  jest.useFakeTimers();

  firstDate = new Date(Date.UTC(2025, 9, 2, 15, 25, 34));
  secondDate = new Date(Date.UTC(2025, 9, 2, 15, 26, 45));
});

beforeEach(() => {
  jest.clearAllMocks();
});

afterAll(() => {
  global.WebSocket = originalWebSocket;
});

describe("Chat", () => {
  it("renders a Chat component with textbox and buttons", () => {
    render(<Chat />);

    const textarea = screen.getByRole("textbox");

    expect(textarea).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Dictate" })).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Ask Question" }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: "Save Memory" }),
    ).toBeInTheDocument();
  });

  it("disables ask and store buttons when textbox is empty", () => {
    render(<Chat />);

    const askButton = screen.getByRole("button", { name: "Ask Question" });
    const storeButton = screen.getByRole("button", { name: "Save Memory" });
    expect(askButton).toBeDisabled();
    expect(storeButton).toBeDisabled();

    const input = screen.getByRole("textbox");
    expect(input).not.toBeDisabled();

    fireEvent.change(input, { target: { value: "Test" } });

    expect(askButton).not.toBeDisabled();
    expect(storeButton).not.toBeDisabled();
  });

  it("connects to memory websocket, disables further input and shows messages when storing memories", async () => {
    jest
      .spyOn(crypto, "randomUUID")
      .mockImplementationOnce(() => "test-uuid")
      .mockImplementationOnce(() => "test-uuid-usermessage");
    jest
      .spyOn(global, "Date")
      .mockImplementationOnce(() => firstDate)
      .mockImplementationOnce(() => secondDate);

    render(<Chat />);

    const input = screen.getByRole("textbox");
    expect(input).toBeInTheDocument();
    expect(input).not.toBeDisabled();
    expect(MessageList).toHaveBeenCalledTimes(1);
    expect(MessageList).toHaveBeenNthCalledWith(1, { messages: [] }, undefined);

    fireEvent.change(input, { target: { value: "Test memory" } });

    const saveButton = screen.getByRole("button", { name: "Save Memory" });
    expect(saveButton).toBeInTheDocument();
    expect(saveButton).not.toBeDisabled();
    expect(mockWebSocketConstructor).not.toHaveBeenCalled();

    fireEvent.click(saveButton);

    expect(mockWebSocketConstructor).toHaveBeenCalledTimes(1);
    expect(mockWebSocketConstructor).toHaveBeenCalledWith(
      "ws://localhost:8080/ws/memory",
    );
    expect(mockWebSocket.send).not.toHaveBeenCalled();

    // Simulate WebSocket open
    act(() => {
      triggerWebSocketEvent("open");
    });

    const expectedMessage = MemoryChunk.encode({
      textData: "Test memory",
      metadata: {
        memoryId: "test-uuid",
        sessionId: "",
        type: ChunkType.MEMORY,
        isFinal: true,
        score: 0,
      },
    }).finish();
    expect(mockWebSocket.send).toHaveBeenCalledTimes(1);
    expect(mockWebSocket.send).toHaveBeenCalledWith(expectedMessage);
    expect(mockWebSocket.close).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Ask Question" })).toBeDisabled();
    expect(saveButton).toBeDisabled();
    expect(input).toBeDisabled();

    const responseMessage = MemoryChunk.encode({
      textData: "Test memory saved with ID: test-uuid",
      metadata: {
        memoryId: "test-uuid",
        sessionId: "",
        type: ChunkType.MEMORY,
        isFinal: true,
        score: 0,
      },
    }).finish();
    act(() => {
      triggerWebSocketEvent("message", responseMessage);
    });

    await waitFor(() => {
      expect(mockWebSocket.close).toHaveBeenCalledTimes(1);
    });
    expect(input).toHaveValue("");
    expect(input).not.toBeDisabled();
    expect(MessageList).toHaveBeenLastCalledWith(
      {
        messages: [
          {
            content: "Test memory",
            id: "test-uuid-usermessage",
            isFinal: true,
            sender: "user",
            timestamp: firstDate,
          },
          {
            content: "Test memory saved with ID: test-uuid",
            id: "test-uuid",
            isFinal: true,
            sender: "assistant",
            timestamp: secondDate,
          },
        ],
      },
      undefined,
    );
  });

  it("connects to ask websocket, disables further input and shows messages when asking questions", async () => {
    jest
      .spyOn(crypto, "randomUUID")
      .mockImplementationOnce(() => "test-qa-uuid")
      .mockImplementationOnce(() => "test-qa-uuid-usermessage");
    jest
      .spyOn(global, "Date")
      .mockImplementationOnce(() => firstDate)
      .mockImplementationOnce(() => secondDate);

    render(<Chat />);

    const input = screen.getByRole("textbox");
    expect(input).toBeInTheDocument();
    expect(input).not.toBeDisabled();
    expect(MessageList).toHaveBeenCalledTimes(1);
    expect(MessageList).toHaveBeenCalledWith({ messages: [] }, undefined);

    fireEvent.change(input, { target: { value: "What did I do?" } });

    const askButton = screen.getByRole("button", { name: "Ask Question" });
    expect(askButton).toBeInTheDocument();
    expect(askButton).not.toBeDisabled();
    expect(mockWebSocketConstructor).not.toHaveBeenCalled();

    fireEvent.click(askButton);

    expect(mockWebSocketConstructor).toHaveBeenCalledTimes(1);
    expect(mockWebSocketConstructor).toHaveBeenCalledWith(
      "ws://localhost:8080/ws/ask",
    );
    expect(mockWebSocket.send).not.toHaveBeenCalled();

    // Simulate WebSocket open
    act(() => {
      triggerWebSocketEvent("open");
    });

    const questionMessage = MemoryChunk.encode({
      textData: "What did I do?",
      metadata: {
        memoryId: "",
        sessionId: "test-qa-uuid",
        type: ChunkType.QUESTION,
        isFinal: true,
        score: 0,
      },
    }).finish();
    expect(mockWebSocket.send).toHaveBeenCalledTimes(1);
    expect(mockWebSocket.send).toHaveBeenCalledWith(questionMessage);
    expect(mockWebSocket.close).not.toHaveBeenCalled();
    expect(screen.getByRole("button", { name: "Save Memory" })).toBeDisabled();
    expect(askButton).toBeDisabled();
    expect(input).toBeDisabled();

    const metadata: ChunkMetadata = {
      sessionId: "test-qa-uuid",
      memoryId: "",
      type: ChunkType.ANSWER,
      isFinal: false,
      score: 0,
    };
    const responseMessages = [
      MemoryChunk.encode({
        textData: "<think>The user asks",
        metadata: {
          ...metadata,
        },
      }).finish(),
      MemoryChunk.encode({
        textData: " what he did",
        metadata: {
          ...metadata,
        },
      }).finish(),
      MemoryChunk.encode({
        textData: " so I shall answer",
        metadata: {
          ...metadata,
        },
      }).finish(),
      MemoryChunk.encode({
        textData: " honestly</think>You did",
        metadata: {
          ...metadata,
        },
      }).finish(),
      MemoryChunk.encode({
        textData: " absolutely nothing.",
        metadata: {
          ...metadata,
          isFinal: true,
          score: 0.96,
        },
      }).finish(),
    ];
    act(() => {
      responseMessages.forEach((message) =>
        triggerWebSocketEvent("message", message),
      );
    });

    await waitFor(() => {
      expect(mockWebSocket.close).toHaveBeenCalledTimes(1);
    });
    expect(input).toHaveValue("");

    expect(input).not.toBeDisabled();
    expect(MessageList).toHaveBeenLastCalledWith(
      {
        messages: [
          {
            content: "What did I do?",
            id: "test-qa-uuid-usermessage",
            isFinal: true,
            sender: "user",
            timestamp: firstDate,
          },
          {
            content:
              "<think>The user asks what he did so I shall answer honestly</think>You did absolutely nothing.",
            id: "test-qa-uuid",
            isFinal: true,
            sender: "assistant",
            timestamp: secondDate,
          },
        ],
      },
      undefined,
    );
  });
});
