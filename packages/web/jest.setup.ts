import "@testing-library/jest-dom";
import { TextEncoder, TextDecoder } from "util";

global.TextEncoder = TextEncoder as typeof global.TextEncoder;
global.TextDecoder = TextDecoder as typeof global.TextDecoder;

// Silence console log and debug messages during tests to reduce noise
jest.spyOn(console, "debug").mockImplementation(() => {});
jest.spyOn(console, "log").mockImplementation(() => {});

Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
  writable: false,
  value: jest.fn(),
});

Object.defineProperty(window, "matchMedia", {
  writable: true,
  value: jest.fn().mockImplementation((query) => ({
    matches: false,
    media: query,
    onchange: null,
    addListener: jest.fn(), // deprecated
    removeListener: jest.fn(), // deprecated
    addEventListener: jest.fn(),
    removeEventListener: jest.fn(),
    dispatchEvent: jest.fn(),
  })),
});
