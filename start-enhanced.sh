#!/bin/bash

# Enhanced UZH Master Project System Startup Script with MS MARCO Integration
# Author: GitHub Copilot  
# Date: August 27, 2025

set -e  # Exit on any error

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Global variables
API_PID=""
WEB_PID=""
EVALUATION_DEMO_PID=""
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Startup mode (simple, marco, marco-light)
STARTUP_MODE="simple"

print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    UZH Master Project                       ║"
    echo "║              Transcription + RAG System                     ║"
    echo "║                                                              ║"
    echo "║  🎤 Real-time Speech-to-Text                                ║"
    echo "║  🧠 Memory-based RAG System                                 ║"
    echo "║  📊 MS MARCO Evaluation (Optional)                          ║"
    echo "║  🌐 Next.js Frontend                                        ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_usage() {
    echo -e "${YELLOW}Usage: $0 [MODE] [OPTIONS]${NC}"
    echo ""
    echo -e "${CYAN}Startup Modes:${NC}"
    echo "  simple       Start with basic RAG (default)"
    echo "  marco        Start with full MS MARCO evaluation"
    echo "  marco-light  Start with lightweight MS MARCO evaluation"
    echo "  demo         Run evaluation demonstration only"
    echo ""
    echo -e "${CYAN}Options:${NC}"
    echo "  --verbose    Enable debug logging"
    echo "  --help       Show this help message"
    echo ""
    echo -e "${CYAN}Examples:${NC}"
    echo "  $0                    # Simple mode"
    echo "  $0 marco --verbose    # Full MS MARCO with debug logs"
    echo "  $0 marco-light        # Lightweight MS MARCO evaluation"
    echo "  $0 demo               # Run evaluation demo only"
}

parse_arguments() {
    VERBOSE_FLAG=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            simple|marco|marco-light|demo)
                STARTUP_MODE="$1"
                shift
                ;;
            --verbose|-v)
                VERBOSE_FLAG="--verbose"
                shift
                ;;
            --help|-h)
                print_usage
                exit 0
                ;;
            *)
                echo -e "${RED}❌ Unknown argument: $1${NC}"
                print_usage
                exit 1
                ;;
        esac
    done
}

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

check_prerequisites() {
    print_status "Checking system prerequisites..."
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js v22+"
        exit 1
    fi
    
    # Check pnpm
    if ! command -v pnpm &> /dev/null; then
        print_error "pnpm is not installed. Please install pnpm"
        exit 1
    fi
    
    # Check Python/uv
    if ! command -v uv &> /dev/null; then
        print_error "uv is not installed. Please install uv Python package manager"
        exit 1
    fi
    
    # Check if virtual environment exists
    if [[ ! -d "$PROJECT_ROOT/.venv" ]]; then
        print_warning "Virtual environment not found. Creating..."
        cd "$PROJECT_ROOT"
        uv venv
    fi
    
    print_status "All prerequisites satisfied!"
}

install_dependencies() {
    print_status "Installing dependencies..."
    
    # Install API dependencies
    cd "$PROJECT_ROOT/packages/api"
    if [[ ! -f "package.json" ]]; then
        print_error "API package.json not found"
        exit 1
    fi
    pnpm install
    
    # Install Python dependencies for API
    source "$PROJECT_ROOT/.venv/bin/activate"
    uv pip install -e .
    
    # Install web dependencies
    cd "$PROJECT_ROOT/packages/web"
    if [[ ! -f "package.json" ]]; then
        print_error "Web package.json not found"
        exit 1
    fi
    pnpm install
    
    # Install evaluation dependencies if using MS MARCO
    if [[ "$STARTUP_MODE" == "marco" || "$STARTUP_MODE" == "marco-light" || "$STARTUP_MODE" == "demo" ]]; then
        print_status "Installing MS MARCO evaluation dependencies..."
        cd "$PROJECT_ROOT/packages/evaluation"
        uv pip install -e .
        # Install additional MS MARCO dependencies
        uv pip install pandas ir-datasets nltk scikit-learn
    fi
    
    print_status "Dependencies installed successfully!"
}

cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down system...${NC}"
    
    if [[ -n "$API_PID" ]]; then
        print_status "Stopping API server (PID: $API_PID)"
        kill $API_PID 2>/dev/null || true
    fi
    
    if [[ -n "$WEB_PID" ]]; then
        print_status "Stopping web server (PID: $WEB_PID)"
        kill $WEB_PID 2>/dev/null || true
    fi
    
    if [[ -n "$EVALUATION_DEMO_PID" ]]; then
        print_status "Stopping evaluation demo (PID: $EVALUATION_DEMO_PID)"
        kill $EVALUATION_DEMO_PID 2>/dev/null || true
    fi
    
    print_status "System shutdown complete"
    exit 0
}

start_api_server() {
    print_status "Starting API server..."
    cd "$PROJECT_ROOT/packages/api"
    source "$PROJECT_ROOT/.venv/bin/activate"
    
    # Determine API startup command based on mode
    case $STARTUP_MODE in
        marco)
            print_status "🔬 Starting API with full MS MARCO evaluation"
            python main.py --marco $VERBOSE_FLAG &
            ;;
        marco-light)
            print_status "🔬 Starting API with lightweight MS MARCO evaluation"
            python main.py --marco-light $VERBOSE_FLAG &
            ;;
        simple|*)
            print_status "📝 Starting API with simple RAG"
            python main.py $VERBOSE_FLAG &
            ;;
    esac
    
    API_PID=$!
    print_status "API server started (PID: $API_PID)"
    
    # Wait for API to be ready
    sleep 3
    if ! kill -0 $API_PID 2>/dev/null; then
        print_error "API server failed to start"
        exit 1
    fi
}

start_web_server() {
    print_status "Starting web frontend..."
    cd "$PROJECT_ROOT/packages/web"
    
    # Start Next.js development server
    pnpm run dev &
    WEB_PID=$!
    print_status "Web server started (PID: $WEB_PID)"
    
    # Wait for web server to be ready
    sleep 5
    if ! kill -0 $WEB_PID 2>/dev/null; then
        print_error "Web server failed to start"
        exit 1
    fi
}

run_evaluation_demo() {
    if [[ "$STARTUP_MODE" == "demo" ]]; then
        print_status "🎯 Running MS MARCO evaluation demonstration..."
        cd "$PROJECT_ROOT/packages/evaluation"
        source "$PROJECT_ROOT/.venv/bin/activate"
        
        python final_pipeline_demo.py &
        EVALUATION_DEMO_PID=$!
        print_status "Evaluation demo started (PID: $EVALUATION_DEMO_PID)"
    fi
}

print_system_info() {
    echo -e "\n${CYAN}🚀 SYSTEM READY!${NC}"
    echo -e "${BLUE}================================${NC}"
    
    case $STARTUP_MODE in
        marco)
            echo -e "${PURPLE}Mode: Full MS MARCO Evaluation${NC}"
            echo -e "${CYAN}Features:${NC}"
            echo "  • Real-time transcription with speech-to-text"
            echo "  • Memory-based RAG with MS MARCO comparison"
            echo "  • Performance metrics (Precision@K, Recall@K, NDCG)"
            echo "  • Industry-standard evaluation benchmarks"
            ;;
        marco-light)
            echo -e "${PURPLE}Mode: Lightweight MS MARCO Evaluation${NC}"
            echo -e "${CYAN}Features:${NC}"
            echo "  • Real-time transcription with speech-to-text"
            echo "  • Memory-based RAG with lightweight MS MARCO"
            echo "  • Basic performance metrics"
            ;;
        demo)
            echo -e "${PURPLE}Mode: Evaluation Demonstration Only${NC}"
            echo -e "${CYAN}Features:${NC}"
            echo "  • MS MARCO dataset analysis"
            echo "  • Evaluation pipeline demonstration"
            echo "  • Performance benchmarking"
            ;;
        simple|*)
            echo -e "${PURPLE}Mode: Simple RAG System${NC}"
            echo -e "${CYAN}Features:${NC}"
            echo "  • Real-time transcription with speech-to-text"
            echo "  • Memory-based RAG with keyword search"
            echo "  • Fast and lightweight operation"
            ;;
    esac
    
    echo -e "\n${CYAN}Access Points:${NC}"
    if [[ "$STARTUP_MODE" != "demo" ]]; then
        echo "  🌐 Web Frontend:  http://localhost:3000"
        echo "  🔗 API Server:    localhost:50051 (gRPC)"
    fi
    echo "  📊 Evaluation:    Check terminal output above"
    
    echo -e "\n${CYAN}Test Commands:${NC}"
    if [[ "$STARTUP_MODE" == "marco" || "$STARTUP_MODE" == "marco-light" ]]; then
        echo "  Test with MS MARCO comparison:"
        echo "  curl -X POST http://localhost:3000/api/test-rag \\"
        echo "       -H 'Content-Type: application/json' \\"
        echo "       -d '{\"query\":\"what is machine learning\",\"compare\":true}'"
    fi
    
    echo "  Simple test:"
    echo "  Open web interface and try recording audio or typing questions"
    
    echo -e "\n${YELLOW}Press Ctrl+C to shutdown all services${NC}"
}

wait_for_shutdown() {
    # Set up signal handlers
    trap cleanup SIGINT SIGTERM
    
    # Wait for user interrupt
    while true; do
        sleep 1
        
        # Check if processes are still running
        if [[ -n "$API_PID" ]] && ! kill -0 $API_PID 2>/dev/null; then
            print_error "API server stopped unexpectedly"
            cleanup
        fi
        
        if [[ -n "$WEB_PID" ]] && ! kill -0 $WEB_PID 2>/dev/null; then
            print_error "Web server stopped unexpectedly"
            cleanup
        fi
    done
}

main() {
    parse_arguments "$@"
    print_banner
    
    echo -e "${CYAN}Starting system in '$STARTUP_MODE' mode...${NC}\n"
    
    check_prerequisites
    install_dependencies
    
    if [[ "$STARTUP_MODE" == "demo" ]]; then
        run_evaluation_demo
        print_system_info
        wait_for_shutdown
    else
        start_api_server
        start_web_server
        run_evaluation_demo  # This will only run if mode is 'demo'
        print_system_info
        wait_for_shutdown
    fi
}

# Run main function with all arguments
main "$@"
