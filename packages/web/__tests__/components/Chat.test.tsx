import "@testing-library/jest-dom";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import Chat from "@/components/Chat";
import { ChunkType, MemoryChunk } from "protos/generated/ts/stt";

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

  it("connects to WebSocket, disables further input and shows messages when storing memories", async () => {
    jest.spyOn(crypto, "randomUUID").mockImplementation(() => "test-uuid");

    render(<Chat />);

    const input = screen.getByRole("textbox");
    expect(input).toBeInTheDocument();
    expect(input).not.toBeDisabled();

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
    expect(
      screen.getByText("Test memory saved with ID: test-uuid"),
    ).toBeInTheDocument();
    expect(screen.getByText("Test memory")).toBeInTheDocument();
    expect(screen.getByText("Test memory")).toBeInstanceOf(HTMLDivElement);

    expect(input).not.toBeDisabled();
  });
});
