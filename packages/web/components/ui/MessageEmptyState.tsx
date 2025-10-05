"use client";

import { FC } from "react";
import { Box, Typography, Stack } from "@mui/joy";
import { Save, Help, MicNone } from "@mui/icons-material";

const MessageEmptyState: FC = () => {
  return (
    <Box
      sx={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        height: "100%",
        textAlign: "center",
        p: 4,
      }}
    >
      <Stack spacing={3} alignItems="center">
        <Box
          sx={{
            display: "flex",
            gap: 2,
            mb: 2,
          }}
        >
          <Save
            sx={{
              fontSize: 48,
              color: "primary.400",
              opacity: 0.8,
            }}
          />
          <Help
            sx={{
              fontSize: 48,
              color: "primary.400",
              opacity: 0.8,
            }}
          />
          <MicNone
            sx={{
              fontSize: 48,
              color: "primary.400",
              opacity: 0.8,
            }}
          />
        </Box>

        <Typography
          level="h3"
          sx={{
            color: "neutral.700",
            mb: 1,
          }}
        >
          Your Memory Assistant
        </Typography>

        <Typography
          level="body-lg"
          sx={{
            color: "neutral.600",
            maxWidth: 500,
            lineHeight: 1.6,
          }}
        >
          Start your conversation by storing memories or asking questions about
          what you&apos;ve shared.
        </Typography>

        <Stack spacing={2} sx={{ mt: 3 }}>
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1.5,
              color: "neutral.600",
            }}
          >
            <Save sx={{ fontSize: 20, color: "primary.500" }} />
            <Typography level="body-md">
              <strong>Save memories</strong> by typing or recording your
              thoughts
            </Typography>
          </Box>

          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1.5,
              color: "neutral.600",
            }}
          >
            <Help sx={{ fontSize: 20, color: "primary.500" }} />
            <Typography level="body-md">
              <strong>Ask questions</strong> to retrieve relevant information
            </Typography>
          </Box>

          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1.5,
              color: "neutral.600",
            }}
          >
            <MicNone sx={{ fontSize: 20, color: "primary.500" }} />
            <Typography level="body-md">
              <strong>Use dictation</strong> for hands-free interaction
            </Typography>
          </Box>
        </Stack>
      </Stack>
    </Box>
  );
};

export default MessageEmptyState;
