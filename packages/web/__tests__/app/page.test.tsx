import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import Page from "../../app/page";

describe("Page", () => {
  it("renders a Next.js image", () => {
    render(<Page />);

    const image = screen.getByRole("img", { name: "Next.js logo" });

    expect(image).toBeInTheDocument();
  });
});
