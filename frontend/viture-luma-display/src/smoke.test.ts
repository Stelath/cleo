import { describe, expect, it } from "vitest";

describe("frontend smoke", () => {
  it("runs under vitest", () => {
    expect(1 + 1).toBe(2);
  });
});
