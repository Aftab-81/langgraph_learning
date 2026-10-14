from mcp.server.fastmcp import FastMCP

# Create the MCP server and give it a name
mcp = FastMCP("MathServer")


@mcp.tool()
async def add(a: float, b: float) -> float:
    """Add two numbers: a + b"""
    return a + b


@mcp.tool()
async def subtract(a: float, b: float) -> float:
    """Subtract b from a: a - b"""
    return a - b


@mcp.tool()
async def multiply(a: float, b: float) -> float:
    """Multiply two numbers: a * b"""
    return a * b


@mcp.tool()
async def divide(a: float, b: float) -> float:
    """Divide a by b: a / b"""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp.tool()
async def modulo(a: float, b: float) -> float:
    """Remainder when a is divided by b: a % b"""
    if b == 0:
        raise ValueError("Cannot take modulo by zero")
    return a % b


@mcp.tool()
async def power(a: float, b: float) -> float:
    """Raise a to the power of b: a ** b"""
    return a ** b


if __name__ == "__main__":
    # Runs the server over stdio (for local use)
    mcp.run()