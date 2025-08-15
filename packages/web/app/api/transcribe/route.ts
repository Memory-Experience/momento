import { NextRequest } from "next/server";
import * as grpc from "@grpc/grpc-js";
import {
  AudioChunk,
  StreamResponse,
  TranscriptionServiceClient,
} from "protos/generated/ts/clients/stt";

// Store active sessions with their gRPC streams
const sessions = new Map<
  string,
  {
    controller: ReadableStreamDefaultController;
    grpcStream: grpc.ClientDuplexStream<AudioChunk, StreamResponse>;
    audioBuffer: Uint8Array[];
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

  if (!sessionId) {
    return new Response("Missing sessionId", { status: 400 });
  }

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

          if (response.transcript) {
            const message = JSON.stringify({
              type: "transcript",
              text: response.transcript.text,
              timestamp: Date.now(),
            });

            controller.enqueue(`data: ${message}\n\n`);
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
          try {
            controller.close();
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
      const audioChunk = {
        data: audioBytes,
      };

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
    // End the gRPC stream gracefully
    if (session.grpcStream && !session.grpcStream.destroyed) {
      session.grpcStream.end();
    }
    sessions.delete(sessionId);
    console.log(`Session ${sessionId} ended`);
  }

  return new Response(JSON.stringify({ success: true }), {
    headers: { "Content-Type": "application/json" },
  });
}
