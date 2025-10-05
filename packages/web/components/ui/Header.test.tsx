import "@testing-library/jest-dom";
import { render, screen } from "@testing-library/react";
import Header from "@/components/ui/Header";

describe("Header", () => {
  it("renders Header component", () => {
    render(<Header />);

    const link = screen.getByRole("link", { name: "Momento Logo" });
    const header = screen.getByRole("img", { name: "Momento Logo" });

    expect(link).toBeInTheDocument();
    expect(link).toHaveAttribute("href", "/");
    expect(header).toBeInTheDocument();
  });
});
