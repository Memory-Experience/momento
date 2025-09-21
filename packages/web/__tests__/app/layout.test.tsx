import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import RootLayout from "../../app/layout";

describe("Layout", () => {
  it("renders its children", () => {
    render(
      <RootLayout>
        <h1>Test</h1>
      </RootLayout>,
      { container: document },
    );

    const heading = screen.getByRole("heading", { name: "Test" });
    const body = heading.parentElement;

    expect(heading).toBeInTheDocument();
    expect(body).not.toBeNull();
    expect(body?.getAttribute("class")).toContain("antialiased");
    expect(body).toBeInstanceOf(HTMLBodyElement);
  });
});
