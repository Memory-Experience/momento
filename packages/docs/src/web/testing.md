# Testing

The Momento web frontend implements comprehensive testing using industry-standard tools and follows best practices for testing React applications. The testing strategy prioritizes user-centric tests that verify behavior rather than implementation details.

## Testing Stack

### React Testing Library

[React Testing Library](https://testing-library.com/docs/react-testing-library/intro/) provides utilities for testing React components following the [Guiding Principles](https://testing-library.com/docs/guiding-principles).

**Core Philosophy**:

- **User-centric queries**: Use accessible queries (roles, labels) over implementation details (test IDs, class names)
- **Behavioral testing**: Test what users see and interact with, not internal state
- **Accessibility focus**: Queries that work with assistive technologies ensure accessible UIs

### Jest

[Jest](https://jestjs.io/) serves as the test runner and assertion library, providing:

- **Fast execution**: Intelligent test parallelization and caching
- **Snapshot testing**: For component output verification
- **Mocking capabilities**: Built-in mocking for modules, functions, and timers
- **Coverage reporting**: Code coverage analysis with v8 provider

Configuration in `jest.config.ts` includes:

- **Test environment**: `jsdom` for DOM simulation
- **Coverage collection**: Comprehensive coverage from all source files
- **Transform**: `ts-jest` for TypeScript support via Next.js jest config
- **Setup**: `jest.setup.ts` for global test configuration

## Testing Patterns

### Component Testing Structure

Tests follow a consistent structure aligned with [React Testing Library best practices](https://testing-library.com/docs/react-testing-library/example-intro):

```typescript
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import "@testing-library/jest-dom";

describe("ComponentName", () => {
  beforeEach(() => {
    // Setup: clear mocks, reset state
  });

  it("describes what the user sees or can do", () => {
    // 1. Arrange: render component
    render(<Component />);

    // 2. Act: simulate user interaction
    fireEvent.click(screen.getByRole("button", { name: "Submit" }));

    // 3. Assert: verify expected outcome
    expect(screen.getByText("Success")).toBeInTheDocument();
  });
});
```

**Naming Convention**: Test descriptions use user-focused language:

- ✅ "disables submit button when form is invalid"
- ✅ "displays error message when submission fails"
- ❌ "sets isValid state to false"
- ❌ "calls handleSubmit callback"

### Mocking

Jest provides powerful mocking capabilities to isolate components and simulate external dependencies.
See [Jest Manual Mocks](https://jestjs.io/docs/manual-mocks) and [Jest ES6 Class Mocks](https://jestjs.io/docs/es6-class-mocks) for detailed documentation.

Some components rely on complex browser APIs or network interactions that require advanced mocking techniques. Momento relies heavily on WebSockets for communication and on browser media APIs for speech-to-text transcription. Examples on how to effectively mock these can be found here:

- **WebSocket Mocking Pattern**: See [`Chat.test.tsx`](https://github.com/Memory-Experience/momento/blob/main/packages/web/components/Chat.test.tsx)
- **Audio Mocking Pattern** (`AudioContext`, `AudioWorklet`, `MediaDevices` and `MediaStream` APIs): See [`AudioRecorder.test.tsx`](https://github.com/Memory-Experience/momento/blob/main/packages/web/components/controls/AudioRecorder.test.tsx)

### Async Testing with act() and waitFor()

React Testing Library requires wrapping state updates in `act()` to ensure all effects complete:

```typescript
// Synchronous state update
act(() => {
    triggerWebSocketEvent("open");
});

// Asynchronous assertion
await waitFor(() => {
    expect(mockWebSocket.close).toHaveBeenCalled();
});
```

**When to use**:

- `act()`: Wraps operations that cause state updates (events, async operations)
- `waitFor()`: Waits for asynchronous updates to complete before asserting

See [React Testing Library's act() documentation](https://testing-library.com/docs/react-testing-library/api/#act) for detailed guidance.

### Fake Timers

Tests use Jest's fake timers to control time-dependent behavior:

```typescript
beforeAll(() => {
    jest.useFakeTimers();
    // Mock Date for consistent timestamps
    jest.spyOn(global, "Date").mockImplementation(() => fixedDate);
});

afterAll(() => {
    jest.useRealTimers();
});
```

**Benefits**:

- Deterministic test execution (no race conditions)
- Instant timeout simulation (`jest.runAllTimers()`)
- Controllable date/time for timestamp testing

### Mock Component Pattern

Complex child components can be mocked to isolate parent component testing:

```typescript
jest.mock("./ui/MessageList", () => jest.fn(() => <></>));
```

This allows testing the `Chat` component's logic without rendering the full `MessageList` component tree. Tests can verify the component is called with correct props:

```typescript
expect(MessageList).toHaveBeenCalledWith(
    {
        messages: [
            { id: "1", content: "Hello", sender: "user" },
            { id: "2", content: "Hi there", sender: "assistant" },
        ],
    },
    undefined, // context
);
```

## Testing Best Practices

### 1. User-Centric Queries

Prefer queries that reflect how users interact with the application:

```typescript
// ✅ Good: accessible query
screen.getByRole("button", { name: "Save Memory" });

// ❌ Avoid: implementation detail
screen.getByTestId("save-button");
```

See [Which query should I use?](https://testing-library.com/docs/queries/about#priority) in Testing Library documentation.

### 2. Avoid Testing Implementation Details

Test behavior, not implementation:

```typescript
// ✅ Good: tests observable behavior
expect(screen.getByText("Memory saved")).toBeInTheDocument();

// ❌ Avoid: tests internal state
expect(component.state.messages).toHaveLength(1);
```

### 3. Mock External Dependencies

Mock browser APIs and network requests:

```typescript
// Mock WebSocket
global.WebSocket = mockWebSocketConstructor;

// Mock MediaRecorder
global.MediaRecorder = MockMediaRecorder;

// Mock crypto
jest.spyOn(crypto, "randomUUID").mockReturnValue("fixed-id");
```

### 4. Use Fake Timers for Time-Dependent Code

Control time for deterministic tests:

```typescript
jest.useFakeTimers();

// Test timeout behavior
jest.advanceTimersByTime(5000);
expect(mockWebSocket.close).toHaveBeenCalled();

jest.useRealTimers();
```

### 5. Clean Up After Tests

Prevent test pollution:

```typescript
beforeEach(() => {
    jest.clearAllMocks();
});

afterAll(() => {
    jest.useRealTimers();
    // Restore original implementations
});
```

## References

- [Jest Documentation](https://jestjs.io/docs/getting-started)
- [React Testing Library Documentation](https://testing-library.com/docs/react-testing-library/intro/)
- [Testing Library Guiding Principles](https://testing-library.com/docs/guiding-principles)
- [Next.js Testing Documentation](https://nextjs.org/docs/app/building-your-application/testing/jest)
