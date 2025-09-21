import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import Chat from "../../components/Chat";

describe("Chat", () => {
  it("renders Chat component", () => {
    render(<Chat />);

    const chat = screen.getByText("Chat component");

    expect(chat).toBeInTheDocument();
  });
});
