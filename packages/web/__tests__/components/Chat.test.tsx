import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import Chat from "@/components/Chat";

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
});
