import { NextRequest } from "next/server";
import * as grpc from "@grpc/grpc-js";
import {
  MemoryChunk,
  ChunkType,
  TranscriptionServiceClient,
} from "protos/generated/ts/clients/stt";

// Store active sessions with their gRPC streams
const sessions = new Map<
  string,
  {
    controller: ReadableStreamDefaultController;
    grpcStream: grpc.ClientDuplexStream<MemoryChunk, MemoryChunk>;
    sessionType: ChunkType;
    memoryId: string;
    isFirstChunk: boolean;
    isClosing: boolean;
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
  const memoryId = searchParams.get("memoryId") || crypto.randomUUID();

  if (!sessionId) {
    return new Response("Missing sessionId", { status: 400 });
  }

  const sessionType =
    sessionTypeParam === "question" ? ChunkType.QUESTION : ChunkType.MEMORY;

  // Check if this is an initialization session
  const isInitSession = sessionId.startsWith("init_");

  // Create Server-Sent Events stream
  const stream = new ReadableStream({
    start(controller) {
      console.log(
        `Starting ${isInitSession ? "initialization" : "recording"} session: ${sessionId}`,
      );

      try {
        // Send immediate connection confirmation
        const connectMessage = JSON.stringify({
          type: "connected",
          sessionId,
          timestamp: Date.now(),
        });

        controller.enqueue(`data: ${connectMessage}\n\n`);

        // For initialization sessions, we don't need a gRPC stream
        if (isInitSession) {
          // Just close the stream after confirmation
          setTimeout(() => controller.close(), 200);
          return;
        }

        // For actual recording sessions, create gRPC stream
        const grpcStream = grpcClient.transcribe();

        // Handle responses from gRPC server
        grpcStream.on("data", (response: MemoryChunk) => {
          if (controller.desiredSize === null) return; // Stream closed

          // Send response to client
          const jsonData = MemoryChunk.toJSON(response);
          controller.enqueue(`data: ${JSON.stringify(jsonData)}\n\n`);
        });

        grpcStream.on("error", (error) => {
          console.error("gRPC error:", error);
          if (controller.desiredSize === null) return; // Stream closed

          controller.enqueue(
            `data: ${JSON.stringify({
              type: "error",
              message: "Backend error occurred",
            })}\n\n`,
          );
        });

        grpcStream.on("end", () => {
          console.log("gRPC stream ended");
          const session = sessions.get(sessionId);
          if (session && !session.isClosing) {
            session.isClosing = true;
            sessions.delete(sessionId);
            if (controller.desiredSize !== null) {
              try {
                controller.close();
              } catch (err) {
                console.warn("Failed to close controller on stream end:", err);
              }
            }
          }
        });

        // Store session
        sessions.set(sessionId, {
          controller,
          grpcStream,
          sessionType,
          memoryId,
          isFirstChunk: true,
          isClosing: false,
        });
      } catch (error) {
        console.error("Error creating session:", error);
        controller.enqueue(
          `data: ${JSON.stringify({
            type: "error",
            message: "Failed to create session",
          })}\n\n`,
        );
        controller.close();
      }
    },

    cancel() {
      console.log(`Canceling session: ${sessionId}`);
      const session = sessions.get(sessionId);
      if (!session) return;

      // Mark as closing to prevent double-close attempts
      session.isClosing = true;

      if (session.grpcStream && !session.grpcStream.destroyed) {
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

// POST handler for sending audio data or final marker
export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const { sessionId, audioData, finalMarker } = body;

    // Validate session ID
    if (!sessionId || typeof sessionId !== "string") {
      return new Response(JSON.stringify({ error: "Invalid session ID" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Get session
    const session = sessions.get(sessionId);
    if (!session) {
      return new Response(JSON.stringify({ error: "Session not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      });
    }

    // Check if this is a final marker request
    if (finalMarker === true) {
      if (session.grpcStream) {
        // Instead of using a magic string, use proper metadata with isFinal flag
        session.grpcStream.write({
          textData: "", // Empty text data is sufficient, no need for magic strings
          metadata: {
            sessionId,
            memoryId: session.memoryId,
            type: session.sessionType,
            isFinal: true, // This is the key change - explicitly mark as final
            score: 0,
          },
        });
        console.log(`Sent final marker for session ${sessionId}`);
        return new Response(JSON.stringify({ success: true }), {
          headers: { "Content-Type": "application/json" },
        });
      }
    }
    // Handle regular audio data
    else if (Array.isArray(audioData)) {
      const audioChunk = {
        audioData: new Uint8Array(audioData),
        metadata: session.isFirstChunk
          ? {
              sessionId,
              memoryId: session.memoryId,
              type: session.sessionType,
              isFinal: false, // Explicitly set to false for normal chunks
              score: 0,
            }
          : undefined,
      };

      session.grpcStream.write(audioChunk);

      // Mark that first chunk was sent
      if (session.isFirstChunk) {
        session.isFirstChunk = false;
      }

      return new Response(JSON.stringify({ success: true }), {
        headers: { "Content-Type": "application/json" },
      });
    } else {
      return new Response(JSON.stringify({ error: "Invalid request body" }), {
        status: 400,
        headers: { "Content-Type": "application/json" },
      });
    }
  } catch (error) {
    console.error("Error processing request:", error);
    return new Response(
      JSON.stringify({ error: "Failed to process request" }),
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

  // For initialization sessions, just return success
  if (sessionId.startsWith("init_")) {
    return new Response(JSON.stringify({ success: true }), {
      headers: { "Content-Type": "application/json" },
    });
  }

  const session = sessions.get(sessionId);
  if (session) {
    // End the gRPC stream
    if (!session.grpcStream.destroyed) {
      session.grpcStream.end();
    }

    // Remove the session
    sessions.delete(sessionId);
  }

  return new Response(JSON.stringify({ success: true }), {
    headers: { "Content-Type": "application/json" },
  });
}
