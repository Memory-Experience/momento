import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import Header from "@/components/ui/Header";

describe("Header", () => {
  it("renders Header component", () => {
    render(<Header />);

    const header = screen.getByRole("heading", { name: "Momento" });

    expect(header).toBeInTheDocument();
  });
});
