import { MemoryChunk } from "protos/generated/ts/stt";

export interface TranscriptionSession {
  sessionId: string;
  eventSource: EventSource | null;
  isConnected: boolean;
  isEnding: boolean;
  type: "memory" | "question";
}

export class TranscriptionService {
  private currentSession: TranscriptionSession | null = null;
  private backendAvailable: boolean = false;
  private isConnecting: boolean = false;
  private hasShownError: boolean = false;
  private errorCallback?: (message: string) => void;
  private connectionStatusCallback?: (isConnected: boolean) => void;
  private messageHandler?: (chunk: MemoryChunk) => void;

  /**
   * Register an error handler to be called when connection errors occur
   */
  public registerErrorHandler(callback: (message: string) => void): void {
    this.errorCallback = callback;
  }

  /**
   * Register a connection status handler
   */
  public registerConnectionStatusHandler(
    callback: (isConnected: boolean) => void,
  ): void {
    this.connectionStatusCallback = callback;
  }

  /**
   * Initialize by establishing a real session to the server
   * This session can then be used later for recording
   */
  public async initialize(): Promise<boolean> {
    console.log("TranscriptionService: Establishing initial session");

    if (this.isConnecting) {
      return this.backendAvailable;
    }

    this.isConnecting = true;

    try {
      // End any existing session first
      if (this.currentSession) {
        await this.endSession();

        // Add a small delay to ensure session is properly closed
        await new Promise((resolve) => setTimeout(resolve, 300));
      }

      // Create a promise that resolves when the connection succeeds or fails
      const connectionPromise = new Promise<boolean>((resolve) => {
        // Start with a memory session type by default
        const success = this.createSession(
          "memory",
          // Dummy message handler - we'll replace this when recording starts
          (chunk) => {
            console.log(
              "TranscriptionService: Received message on initial session:",
              chunk,
            );
            // If we get any messages on the initial session, store them temporarily
            if (this.messageHandler) {
              this.messageHandler(chunk);
            }
          },
          // Status handler for the initial connection
          (connected, error) => {
            if (connected) {
              console.log(
                "TranscriptionService: Initial session connected successfully",
              );
              this.updateBackendStatus(true);
              resolve(true);
            } else {
              console.error(
                `TranscriptionService: Initial session connection failed: ${error}`,
              );
              this.updateBackendStatus(false);
              resolve(false);
            }
          },
        );

        // If session creation failed immediately
        if (!success) {
          resolve(false);
        }
      });

      // Wait for connection result
      const connected = await connectionPromise;
      this.isConnecting = false;
      return connected;
    } catch (error) {
      console.error(
        "TranscriptionService: Error establishing initial session:",
        error,
      );
      this.updateBackendStatus(false);
      this.isConnecting = false;
      return false;
    }
  }

  /**
   * Update backend availability status and notify via callback
   */
  private updateBackendStatus(available: boolean): void {
    const statusChanged = this.backendAvailable !== available;
    this.backendAvailable = available;

    if (available) {
      this.hasShownError = false;
    } else if (statusChanged && this.errorCallback && !this.hasShownError) {
      this.hasShownError = true;
      this.errorCallback("Backend service is unavailable.");
    }

    if (statusChanged && this.connectionStatusCallback) {
      this.connectionStatusCallback(available);
    }
  }

  /**
   * Create or switch to a recording session
   * If we already have a connected session, we'll update its message handler
   */
  public async startRecordingSession(
    type: "memory" | "question",
    onMessage: (chunk: MemoryChunk) => void,
    onSessionStatus: (connected: boolean, error?: string) => void,
  ): Promise<boolean> {
    console.log(`TranscriptionService: Starting ${type} recording session`);

    // Store the message handler for future messages
    this.messageHandler = onMessage;

    // Check if backend is available
    if (!this.backendAvailable) {
      onSessionStatus(
        false,
        "Backend service is unavailable. Please try again later.",
      );
      return false;
    }

    // If we already have a connected session
    if (this.currentSession?.isConnected) {
      // If type matches, we can reuse the session
      if (this.currentSession.type === type) {
        console.log(`TranscriptionService: Reusing existing ${type} session`);
        onSessionStatus(true);
        return true;
      } else {
        // If type doesn't match, end current session and create a new one
        console.log(
          `TranscriptionService: Switching from ${this.currentSession.type} to ${type} session`,
        );

        // Await the end of the current session to prevent race conditions
        await this.endSession();
      }
    }

    // Create a new session
    return this.createSession(type, onMessage, onSessionStatus);
  }

  /**
   * Create a new session with the backend
   */
  private createSession(
    type: "memory" | "question",
    onMessage: (chunk: MemoryChunk) => void,
    onSessionStatus: (connected: boolean, error?: string) => void,
  ): boolean {
    // Generate session ID
    const sessionId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
    console.log(
      `TranscriptionService: Creating new ${type} session with ID ${sessionId}`,
    );

    // Ensure any previous session is properly cleaned up
    if (this.currentSession?.eventSource) {
      this.currentSession.eventSource.close();
      this.currentSession = null;
    }

    // Start Server-Sent Events connection
    const eventSource = new EventSource(
      `/api/transcribe?sessionId=${sessionId}&type=${type}`,
    );

    const session: TranscriptionSession = {
      sessionId,
      eventSource,
      isConnected: false,
      isEnding: false,
      type,
    };
    this.currentSession = session;

    // Set up session timeout
    const sessionTimeout = setTimeout(() => {
      if (!session.isConnected) {
        console.error(
          `TranscriptionService: Session timeout - no confirmation for session ${sessionId}`,
        );
        onSessionStatus(
          false,
          "Session connection timed out. Please try again.",
        );
        this.closeSession();
        // Update global status since we can't connect
        this.updateBackendStatus(false);
      }
    }, 10000);

    // Handle SSE events
    eventSource.onmessage = (event) => {
      try {
        console.log(event.data);
        const data = JSON.parse(event.data);

        if (data.type === "connected") {
          session.isConnected = true;
          clearTimeout(sessionTimeout);
          console.log(
            `TranscriptionService: Session ${sessionId} connected successfully`,
          );
          // Update global status since connection is working
          this.updateBackendStatus(true);
          onSessionStatus(true);
        } else if (data.type === "error") {
          clearTimeout(sessionTimeout);
          console.error(
            `TranscriptionService: Error in session ${sessionId}:`,
            data.message,
          );
          onSessionStatus(false, data.message || "Unknown error from backend");
          this.closeSession();
        } else {
          // Process regular message
          const chunk = MemoryChunk.fromJSON(data);
          onMessage(chunk);
        }
      } catch (error) {
        console.error("TranscriptionService: Error parsing message:", error);
      }
    };

    eventSource.onerror = (error) => {
      clearTimeout(sessionTimeout);
      console.error(
        `TranscriptionService: SSE error in session ${sessionId}:`,
        error,
      );

      if (eventSource.readyState === EventSource.CLOSED) {
        console.log(`TranscriptionService: Session ${sessionId} closed`);
      } else {
        onSessionStatus(false, "Connection to backend lost. Please try again.");
        this.closeSession();
        // Update global status since connection failed
        this.updateBackendStatus(false);
      }
    };

    return true;
  }

  /**
   * Send audio data to the server using the current session
   */
  public async sendAudioData(audioData: Uint8Array): Promise<boolean> {
    if (!this.currentSession) return false;

    try {
      if (!this.currentSession.isConnected) {
        console.warn(
          "TranscriptionService: Attempted to send audio data before session is connected",
        );
        return false;
      }

      // Skip sending audio data if the session is ending
      if (this.currentSession.isEnding || !this.hasActiveSession()) {
        console.log(
          "TranscriptionService: Ignoring audio data - session is ending or has ended",
        );
        return false;
      }

      const response = await fetch("/api/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId: this.currentSession.sessionId,
          audioData: Array.from(audioData),
        }),
      });

      if (!response.ok) {
        console.error(
          "TranscriptionService: Error sending audio data:",
          response.status,
        );
        return false;
      }

      return true;
    } catch (err) {
      console.error("TranscriptionService: Error sending audio data:", err);
      return false;
    }
  }

  /**
   * Send a final marker to indicate end of audio stream
   */
  public async sendFinalMarker(): Promise<boolean> {
    if (!this.currentSession) return false;

    try {
      if (!this.currentSession.isConnected) {
        console.warn(
          "TranscriptionService: Attempted to send final marker before session is connected",
        );
        return false;
      }

      // Skip if the session is ending or has ended
      if (this.currentSession.isEnding || !this.hasActiveSession()) {
        console.log(
          "TranscriptionService: Ignoring final marker - session is ending or has ended",
        );
        return false;
      }

      // Use the existing transcribe endpoint, but with a special finalMarker flag
      const response = await fetch("/api/transcribe", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionId: this.currentSession.sessionId,
          finalMarker: true,
        }),
      });

      if (!response.ok) {
        console.error(
          "TranscriptionService: Error sending final marker:",
          response.status,
        );
        return false;
      }

      console.log("TranscriptionService: Final marker sent successfully");
      return true;
    } catch (err) {
      console.error("TranscriptionService: Error sending final marker:", err);
      return false;
    }
  }

  public closeEventSourceConnection(): void {
    if (this.currentSession?.eventSource) {
      console.log(
        `TranscriptionService: Directly closing EventSource for session ${this.currentSession.sessionId}`,
      );
      this.currentSession.eventSource.close();
    }
  }

  /**
   * End the current session
   */
  public async endSession(): Promise<void> {
    if (!this.currentSession) return;

    const sessionId = this.currentSession.sessionId;
    const sessionType = this.currentSession.type;

    console.log(
      `TranscriptionService: Ending ${sessionType} session ${sessionId}`,
    );

    try {
      // Mark the session as ending first
      this.currentSession.isEnding = true;

      // Close the EventSource first to prevent any new events
      if (this.currentSession.eventSource) {
        this.currentSession.eventSource.close();
      }

      // Now send the DELETE request
      await fetch(`/api/transcribe?sessionId=${sessionId}`, {
        method: "DELETE",
      });

      console.log(
        `TranscriptionService: Successfully ended session ${sessionId}`,
      );
    } catch (error) {
      console.info(
        `TranscriptionService: Error ending session ${sessionId}:`,
        error,
      );
    } finally {
      this.closeSession();
    }
  }

  /**
   * Check if the backend is available
   */
  public isBackendAvailable(): boolean {
    return this.backendAvailable;
  }

  /**
   * Check if there is an active session
   */
  public hasActiveSession(): boolean {
    return this.currentSession !== null;
  }

  /**
   * Check if the active session is connected
   */
  public isSessionConnected(): boolean {
    return !!this.currentSession?.isConnected;
  }

  /**
   * Reset the connection state to allow new connection attempts
   */
  public reset(): void {
    this.hasShownError = false;
    this.isConnecting = false;
    this.messageHandler = undefined;

    if (this.currentSession) {
      this.closeSession();
    }
  }

  private closeSession(): void {
    if (this.currentSession?.eventSource) {
      console.log(
        `TranscriptionService: Closing EventSource for session ${this.currentSession.sessionId}`,
      );
      this.currentSession.eventSource.close();
    }

    if (this.currentSession) {
      console.log(
        `TranscriptionService: Session ${this.currentSession.sessionId} closed`,
      );
      this.currentSession = null;
    }
  }
}

export default TranscriptionService;
