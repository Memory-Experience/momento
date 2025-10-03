import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import MessageEmptyState from "@/components/ui/MessageEmptyState";

describe("MessageEmptyState", () => {
  it("renders the component elements correctly", () => {
    render(<MessageEmptyState />);

    const heading = screen.getByRole("heading", {
      name: "Your Memory Assistant",
    });
    expect(heading).toBeInTheDocument();

    const description = screen.getByText(
      "Start your conversation by storing memories or asking questions about what you've shared.",
    );
    expect(description).toBeInTheDocument();

    const saveMemories = screen.getByText("Save memories");
    expect(saveMemories).toBeInTheDocument();

    const askQuestions = screen.getByText("Ask questions");
    expect(askQuestions).toBeInTheDocument();

    const voiceRecording = screen.getByText("Use dictation");
    expect(voiceRecording).toBeInTheDocument();
  });
});
