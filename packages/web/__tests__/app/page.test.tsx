import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import Page from "../../app/page";
import Chat from "../../components/Chat";

jest.mock("../../components/Chat", () => {
  return jest.fn(() => <h1>Chat Component</h1>);
});

describe("Page", () => {
  it("renders Chat component", () => {
    render(<Page />);

    const chatComponent = screen.getByRole("heading", {
      name: "Chat Component",
    });

    expect(chatComponent).toBeInTheDocument();
    expect(Chat).toHaveBeenCalledTimes(1);
  });
});
