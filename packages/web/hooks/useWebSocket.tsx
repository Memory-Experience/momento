import { ChunkType, MemoryChunk } from "protos/generated/ts/stt";
import { useState, useRef, useCallback, useEffect } from "react";

export const useWebSocket = (url: string | null) => {
  const [isConnected, setIsConnected] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);
  const socketClosingTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  const closeImmediately = useCallback(() => {
    console.debug(`Closing WebSocket for "${url}".`);
    if (socketClosingTimeoutRef.current) {
      clearTimeout(socketClosingTimeoutRef.current);
      socketClosingTimeoutRef.current = null;
    }
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }
    setIsConnected(false);
  }, [url]);

  const connect = useCallback(() => {
    if (!url) {
      console.warn("WebSocket URL is null. Cannot connect.");
      return;
    }
    if (socketRef.current) {
      console.warn("WebSocket is already connected.");
      return;
    }
    socketRef.current = new WebSocket(url);

    socketRef.current.addEventListener("open", () => {
      console.log(`WebSocket connection to ${url} established`);
      setIsConnected(true);
    });

    socketRef.current.addEventListener("close", () => {
      console.log("WebSocket connection closed");
      setIsConnected(false);
      socketRef.current = null;
    });

    socketRef.current.addEventListener(
      "message",
      async (event) => {
        console.debug("useWebSocket: event received", event);
        const data =
          event.data instanceof Blob ? await event.data.bytes() : event.data;
        if (data) {
          const message = MemoryChunk.decode(new Uint8Array(data));
          if (message.metadata?.isFinal) {
            // Close after a small delay to allow other handlers to process
            setTimeout(() => closeImmediately(), 0);
          }
        }
      },
      { passive: true },
    );

    socketRef.current.addEventListener("error", (error) => {
      console.error("WebSocket error: ", error);
      setIsConnected(false);
      socketRef.current = null;
    });
  }, [closeImmediately, url]);

  const disconnect = useCallback(() => {
    if (!socketRef.current) {
      console.warn("WebSocket is not connected.");
      return;
    }

    socketRef.current.send(
      MemoryChunk.encode({
        metadata: {
          sessionId: "",
          memoryId: "",
          type: ChunkType.TRANSCRIPT,
          isFinal: true,
          score: 0,
        },
      }).finish(),
    );

    socketClosingTimeoutRef.current = setTimeout(() => {
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
      setIsConnected(false);
    }, 5000);
  }, []);

  const send = useCallback<typeof WebSocket.prototype.send>((data) => {
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(data);
    } else {
      console.warn("WebSocket is not connected. Unable to send message.");
    }
  }, []);

  const addEventListener = useCallback<
    <K extends keyof WebSocketEventMap>(
      key: K,
      listener: (this: WebSocket, ev: WebSocketEventMap[K]) => void,
      options?: boolean | AddEventListenerOptions,
    ) => boolean
  >((type, listener, options) => {
    if (socketRef.current) {
      console.debug(`Adding event listener of type "${type}" to WebSocket.`);
      socketRef.current.addEventListener(type, listener, options);
      return true;
    }
    console.warn(
      `WebSocket is not connected. Unable to add event listener of type "${type}".`,
    );
    return false;
  }, []);

  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
        socketRef.current = null;
      }
    };
  }, []);

  return { isConnected, addEventListener, connect, disconnect, send };
};
