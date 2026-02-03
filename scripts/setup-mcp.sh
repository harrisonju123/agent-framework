#!/bin/bash
# Setup script for MCP servers

set -e

echo "=== MCP Server Setup ==="
echo ""

# Check Node.js installation
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+ first."
    echo "   Visit: https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node --version | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo "❌ Node.js version $NODE_VERSION found, but version 18+ is required."
    echo "   Please upgrade Node.js: https://nodejs.org/"
    exit 1
fi

echo "✅ Node.js $(node --version) found"
echo ""

# Install JIRA MCP server
echo "📦 Installing JIRA MCP server..."
cd mcp-servers/jira
npm install
npm run build
cd ../..
echo "✅ JIRA MCP server installed"
echo ""

# Install GitHub MCP server
echo "📦 Installing GitHub MCP server..."
cd mcp-servers/github
npm install
npm run build
cd ../..
echo "✅ GitHub MCP server installed"
echo ""

# Copy MCP config if not exists
if [ -f config/mcp-config.json ]; then
    echo "✅ config/mcp-config.json already exists"
    echo ""
else
    if [ -f config/mcp-config.json.example ]; then
        echo "📝 Creating MCP configuration..."
        cp config/mcp-config.json.example config/mcp-config.json
        echo "✅ Created config/mcp-config.json from example"
        echo ""
    else
        echo "⚠️  config/mcp-config.json not found and no example available"
        echo ""
    fi
fi

# Create logs directory
mkdir -p logs

echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Ensure environment variables are set:"
echo "   source scripts/setup-env.sh"
echo ""
echo "2. Enable MCPs in config/agent-framework.yaml:"
echo "   llm:"
echo "     mode: claude_cli"
echo "     use_mcp: true"
echo "     mcp_config_path: \${PWD}/config/mcp-config.json"
echo ""
echo "3. Test MCP servers:"
echo "   cd mcp-servers/jira && npm start  # Ctrl+C to exit"
echo "   cd mcp-servers/github && npm start  # Ctrl+C to exit"
echo ""
echo "4. Start agents with MCP enabled:"
echo "   agent start"
echo ""
echo "For more information, see docs/MCP_SETUP.md"
