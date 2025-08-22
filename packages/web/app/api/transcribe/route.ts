import { NextRequest } from "next/server";
import * as grpc from "@grpc/grpc-js";
import {
  AudioChunk,
  StreamResponse,
  TranscriptionServiceClient,
  SessionType,
} from "protos/generated/ts/clients/stt";

// Store active sessions with their gRPC streams
const sessions = new Map<
  string,
  {
    controller: ReadableStreamDefaultController;
    grpcStream: grpc.ClientDuplexStream<AudioChunk, StreamResponse>;
    audioBuffer: Uint8Array[];
    sessionType: SessionType;
    isFirstChunk: boolean;
  }
>();

// Create gRPC client
const grpcClient = new TranscriptionServiceClient(
  "localhost:50051",
  grpc.credentials.createInsecure(),
);

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const sessionId = searchParams.get("sessionId");
  const sessionTypeParam = searchParams.get("type");

  if (!sessionId) {
    return new Response("Missing sessionId", { status: 400 });
  }

  const sessionType =
    sessionTypeParam === "question" ? SessionType.QUESTION : SessionType.MEMORY;

  // Create Server-Sent Events stream
  const stream = new ReadableStream({
    start(controller) {
      console.log(`Starting transcription session: ${sessionId}`);

      try {
        // Create bidirectional gRPC stream
        const grpcStream = grpcClient.transcribe();

        // Handle responses from gRPC server
        grpcStream.on("data", (response: StreamResponse) => {
          console.log("Received gRPC response:", response);

          // Check if controller is still open before trying to enqueue
          if (controller.desiredSize === null) {
            console.log("Controller already closed, ignoring response");
            return;
          }

          if (response.transcript) {
            const message = JSON.stringify({
              type: "transcript",
              text: response.transcript.text,
              timestamp: Date.now(),
            });

            try {
              controller.enqueue(`data: ${message}\n\n`);
            } catch (error) {
              console.log("Failed to enqueue transcript message:", error);
            }
          } else if (response.answer) {
            const message = JSON.stringify({
              type: "answer",
              text: response.answer.text,
              timestamp: Date.now(),
            });

            try {
              controller.enqueue(`data: ${message}\n\n`);
              console.log("Answer sent, closing connection");
              // Close the controller after sending the answer
              controller.close();
            } catch (error) {
              console.log("Failed to enqueue answer message:", error);
            }
          }
        });

        grpcStream.on("error", (error: Error) => {
          console.error("gRPC stream error:", error);
          const message = JSON.stringify({
            type: "error",
            message: error.message,
            timestamp: Date.now(),
          });
          controller.enqueue(`data: ${message}\n\n`);
        });

        grpcStream.on("end", () => {
          console.log("gRPC stream ended");
          // Clean up the session
          sessions.delete(sessionId);
          try {
            if (controller.desiredSize !== null) {
              controller.close();
            }
          } catch (error: unknown) {
            if (
              !(
                error instanceof Error &&
                error.message.includes("already closed")
              )
            ) {
              console.error("Error closing controller:", error);
            }
          }
        });

        // Store the session
        sessions.set(sessionId, {
          controller,
          grpcStream,
          audioBuffer: [],
          sessionType,
          isFirstChunk: true,
        });

        // Send initial connection message
        const connectMessage = JSON.stringify({
          type: "connected",
          sessionId,
          timestamp: Date.now(),
        });
        controller.enqueue(`data: ${connectMessage}\n\n`);
      } catch (error) {
        console.error("Error setting up gRPC stream:", error);
        const errorMessage = JSON.stringify({
          type: "error",
          message: "Failed to connect to transcription service",
          timestamp: Date.now(),
        });
        controller.enqueue(`data: ${errorMessage}\n\n`);
        controller.close();
      }
    },
    cancel() {
      // Cleanup when client disconnects
      console.log(`Cleaning up session: ${sessionId}`);
      const session = sessions.get(sessionId);
      if (session?.grpcStream) {
        session.grpcStream.end();
      }
      sessions.delete(sessionId);
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache",
      Connection: "keep-alive",
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Headers": "Cache-Control",
    },
  });
}

export async function POST(request: NextRequest) {
  try {
    const { sessionId, audioData } = await request.json();

    if (!sessionId || !audioData) {
      return new Response("Missing sessionId or audioData", { status: 400 });
    }

    const session = sessions.get(sessionId);
    if (!session) {
      return new Response("Session not found", { status: 404 });
    }

    // Convert audio data back to Uint8Array
    const audioBytes = new Uint8Array(audioData);

    // Send audio chunk to gRPC server
    if (session.grpcStream && !session.grpcStream.destroyed) {
      const audioChunk: AudioChunk = {
        data: audioBytes,
        metadata: session.isFirstChunk
          ? {
              sessionId,
              type: session.sessionType,
            }
          : undefined,
      };

      // Mark that we've sent the first chunk
      if (session.isFirstChunk) {
        session.isFirstChunk = false;
      }

      // console.debug(
      //   `Sending audio chunk: ${audioBytes.length} bytes for session ${sessionId}`,
      // );
      session.grpcStream.write(audioChunk);
    } else {
      console.error("gRPC stream not available for session:", sessionId);
      return new Response("gRPC stream not available", { status: 500 });
    }

    return new Response(
      JSON.stringify({
        success: true,
        bytesReceived: audioBytes.length,
      }),
      {
        headers: { "Content-Type": "application/json" },
      },
    );
  } catch (error) {
    console.error("Error processing audio data:", error);
    return new Response(
      JSON.stringify({
        success: false,
        error: "Failed to process audio data",
      }),
      {
        status: 500,
        headers: { "Content-Type": "application/json" },
      },
    );
  }
}

export async function DELETE(request: NextRequest) {
  const { searchParams } = new URL(request.url);
  const sessionId = searchParams.get("sessionId");

  if (!sessionId) {
    return new Response("Missing sessionId", { status: 400 });
  }

  const session = sessions.get(sessionId);
  if (session) {
    // For question sessions, we need to wait for the answer before fully closing
    if (session.sessionType === SessionType.QUESTION) {
      console.log(
        `Question session ${sessionId} - ending audio stream, waiting for answer`,
      );
      // End the gRPC stream to signal no more audio is coming
      if (session.grpcStream && !session.grpcStream.destroyed) {
        session.grpcStream.end();
      }
      // Don't delete the session yet - let the answer handler close the controller
    } else {
      // For memory sessions, close immediately
      console.log(`Memory session ${sessionId} ended`);
      if (session.grpcStream && !session.grpcStream.destroyed) {
        session.grpcStream.end();
      }
      try {
        session.controller.close();
      } catch (error) {
        console.log("Controller already closed:", error);
      }
      sessions.delete(sessionId);
    }
  }

  return new Response(JSON.stringify({ success: true }), {
    headers: { "Content-Type": "application/json" },
  });
}
