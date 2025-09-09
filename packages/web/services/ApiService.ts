/**
 * Backend API client
 * Central place for all backend API endpoints
 */

/**
 * BackendApi provides methods for interacting with the backend service
 * This follows Next.js best practices for API access
 */
export const ApiService = {
  /**
   * Get the URL for SSE connection with the backend
   */
  getStreamUrl(sessionId: string, type: "memory" | "question"): string {
    // In Next.js, relative URLs automatically point to API routes
    return `/api?sessionId=${sessionId}&type=${type}`;
  },

  /**
   * Send audio data to the backend for processing
   */
  async sendAudioData(
    sessionId: string,
    audioData: Uint8Array,
  ): Promise<Response> {
    return fetch("/api", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId,
        audioData: Array.from(audioData),
      }),
    });
  },

  /**
   * Send a final marker to indicate end of audio stream
   */
  async sendFinalMarker(sessionId: string): Promise<Response> {
    return fetch("/api", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId,
        finalMarker: true,
      }),
    });
  },

  /**
   * End a backend session
   */
  async endSession(sessionId: string): Promise<Response> {
    return fetch(`/api?sessionId=${sessionId}`, {
      method: "DELETE",
    });
  },
};

export default ApiService;
