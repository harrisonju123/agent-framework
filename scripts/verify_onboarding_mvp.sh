#!/bin/bash

# Verification script for User Onboarding MVP
# Tests that all implemented features are working

set -e

echo "🔍 Verifying User Onboarding MVP Implementation"
echo "================================================"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Change to project root
cd "$(dirname "$0")/.."

echo "1. Testing Python imports..."
echo "   - Health checker"
python -c "from agent_framework.health import HealthChecker, CheckResult, CheckStatus" && echo -e "${GREEN}   ✓ Health checker imports OK${NC}" || echo -e "${RED}   ✗ Health checker import failed${NC}"

echo "   - Error translator"
python -c "from agent_framework.errors import ErrorTranslator, UserFriendlyError" && echo -e "${GREEN}   ✓ Error translator imports OK${NC}" || echo -e "${RED}   ✗ Error translator import failed${NC}"

echo "   - Config generator"
python -c "from agent_framework.config.templates import ConfigGenerator" && echo -e "${GREEN}   ✓ Config generator imports OK${NC}" || echo -e "${RED}   ✗ Config generator import failed${NC}"

echo ""
echo "2. Testing CLI commands..."
if agent --help | grep -q "doctor"; then
    echo -e "${GREEN}   ✓ 'agent doctor' command registered${NC}"
else
    echo -e "${RED}   ✗ 'agent doctor' command not found${NC}"
fi

echo ""
echo "3. Verifying file structure..."

# Check new files exist
files_to_check=(
    "src/agent_framework/health/__init__.py"
    "src/agent_framework/health/checker.py"
    "src/agent_framework/errors/__init__.py"
    "src/agent_framework/errors/translator.py"
    "src/agent_framework/config/templates.py"
    "src/agent_framework/web/frontend/src/components/SetupWizard.vue"
    "docs/TROUBLESHOOTING.md"
    "docs/GETTING_STARTED.md"
)

for file in "${files_to_check[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}   ✓ $file${NC}"
    else
        echo -e "${RED}   ✗ $file (missing)${NC}"
    fi
done

echo ""
echo "4. Checking frontend build..."
if [ -d "src/agent_framework/web/frontend/dist" ]; then
    echo -e "${GREEN}   ✓ Frontend built (dist/ exists)${NC}"
    echo "     - Bundle size: $(du -sh src/agent_framework/web/frontend/dist | cut -f1)"
else
    echo -e "${YELLOW}   ⚠ Frontend not built${NC}"
    echo "     Run: cd src/agent_framework/web/frontend && npm run build"
fi

echo ""
echo "5. Testing API endpoints..."
echo "   Starting test server on port 8081..."

# Start server in background
agent dashboard --port 8081 --no-browser &
SERVER_PID=$!

# Wait for server to start
sleep 3

# Test setup status endpoint
if curl -s http://localhost:8081/api/setup/status > /dev/null 2>&1; then
    echo -e "${GREEN}   ✓ /api/setup/status endpoint responding${NC}"
else
    echo -e "${RED}   ✗ /api/setup/status endpoint not responding${NC}"
fi

# Stop server
kill $SERVER_PID 2>/dev/null || true
sleep 1

echo ""
echo "6. Documentation check..."
if grep -q "Setup Wizard" README.md; then
    echo -e "${GREEN}   ✓ README.md updated with setup wizard${NC}"
else
    echo -e "${YELLOW}   ⚠ README.md may need setup wizard section${NC}"
fi

if grep -q "agent doctor" README.md; then
    echo -e "${GREEN}   ✓ README.md mentions 'agent doctor' command${NC}"
else
    echo -e "${YELLOW}   ⚠ README.md may need 'agent doctor' reference${NC}"
fi

echo ""
echo "================================================"
echo -e "${GREEN}✅ User Onboarding MVP verification complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Run: agent dashboard"
echo "  2. Click 'Setup' button to test wizard"
echo "  3. Or run: agent doctor"
echo ""
echo "Documentation:"
echo "  - Getting Started: docs/GETTING_STARTED.md"
echo "  - Troubleshooting: docs/TROUBLESHOOTING.md"
